#include "lane_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <opencv2/imgproc.hpp>

namespace lane_pilot
{

namespace
{

// Lateral runs of line pixels in a single bird's-eye row -> their center cols.
void row_clusters(
  const uchar * p, int w, int min_px, int max_px, std::vector<double> & out)
{
  out.clear();
  int c = 0;
  while (c < w) {
    if (p[c] == 0) {++c; continue;}
    int c0 = c;
    while (c < w && p[c] != 0) {++c;}
    const int width = c - c0;          // run is [c0, c)
    if (width >= min_px && width <= max_px) {
      out.push_back((c0 + (c - 1)) * 0.5);
    }
  }
}

}  // namespace

DetectionResult LaneDetector::detect(
  const cv::Mat & bird, const Ipm & ipm, bool keep_debug)
{
  DetectionResult res;
  if (bird.empty()) {return res;}

  cv::Mat gray;
  if (bird.channels() == 3) {
    cv::cvtColor(bird, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bird;
  }
  if (params_.blur_ksize >= 3) {
    const int k = params_.blur_ksize | 1;   // force odd
    cv::GaussianBlur(gray, gray, cv::Size(k, k), 0);
  }

  cv::Mat mask;
  cv::threshold(
    gray, mask, static_cast<double>(params_.black_threshold), 255.0,
    params_.invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
  // The bird's-eye warp fills out-of-FOV and blind-spot pixels with 0 (black);
  // with an inverted threshold those would read as "line". Exclude them, or the
  // tracer latches onto the edge of the no-data wedge instead of a real line.
  if (params_.invert) {
    mask.setTo(0, gray == 0);
  }
  // Optional morphology to clean noise: OPEN removes speckle, CLOSE fills gaps.
  if (params_.open_px >= 1) {
    cv::Mat k = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(params_.open_px, params_.open_px));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
  }
  if (params_.close_px >= 1) {
    cv::Mat k = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(params_.close_px, params_.close_px));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
  }

  const int W = mask.cols;
  const int Hh = mask.rows;
  const double mpp = ipm.view().mpp;
  const double max_jump_px = params_.max_jump_m / std::max(1e-6, mpp);

  // ROI bounds (bird pixels) from the metric ROI — search only this ground band.
  double cdum, r_a, r_b, col_left, col_right, rdum;
  ipm.ground_to_bird(params_.roi_x_max, 0.0, cdum, r_a);
  ipm.ground_to_bird(params_.roi_x_min, 0.0, cdum, r_b);
  ipm.ground_to_bird(0.0, params_.roi_y_half, col_left, rdum);
  ipm.ground_to_bird(0.0, -params_.roi_y_half, col_right, rdum);
  const int roi_row_top = std::clamp(
    static_cast<int>(std::floor(std::min(r_a, r_b))), 0, Hh - 1);
  const int roi_row_bot = std::clamp(
    static_cast<int>(std::ceil(std::max(r_a, r_b))), 0, Hh - 1);
  if (col_left > col_right) {std::swap(col_left, col_right);}

  double current_col = -1.0;            // <0 until the track is acquired
  int rows_with_clusters = 0;
  std::vector<double> clusters;

  static int g_dbg_frame = 0;
  const bool dbg = (std::getenv("LANE_DEBUG") != nullptr) && (g_dbg_frame++ % 20 == 0);
  int dbg_printed = 0;
  if (dbg) {
    std::fprintf(
      stderr, "[LANE_DBG] W=%d Hh=%d mpp=%.4f maxjump_px=%.1f thr=%d "
      "minpx=%d maxpx=%d\n", W, Hh, mpp, max_jump_px, params_.black_threshold,
      params_.min_cluster_px, params_.max_cluster_px);
  }

  // March from the nearest visible row (bottom) to the farthest (top).
  for (int r = Hh - 1; r >= 0; r -= std::max(1, params_.row_step)) {
    if (r < roi_row_top || r > roi_row_bot) {continue;}     // forward ROI
    const uchar * p = mask.ptr<uchar>(r);
    row_clusters(
      p, W, params_.min_cluster_px, params_.max_cluster_px, clusters);
    clusters.erase(                                         // lateral ROI
      std::remove_if(clusters.begin(), clusters.end(),
        [&](double c) {return c < col_left || c > col_right;}),
      clusters.end());
    if (clusters.empty()) {continue;}
    ++rows_with_clusters;

    if (dbg && dbg_printed < 8) {
      std::string s;
      for (double cc : clusters) {char b[16]; std::snprintf(b, 16, "%.0f ", cc); s += b;}
      std::fprintf(stderr, "[LANE_DBG] r=%d cur=%.1f clusters=[%s]\n",
        r, current_col, s.c_str());
      ++dbg_printed;
    }

    double picked = -1.0;
    if (current_col < 0.0) {
      // Seed on the middle line: the cluster nearest image center (Y=0) — but
      // REJECT a seed farther than one max-jump from center. The robot follows
      // the middle line, so the middle line is always the cluster nearest Y=0;
      // the borders sit ~half a lane away and are rejected by the gate. We seed
      // at center EVERY frame (not from a remembered column): a remembered seed
      // can latch onto a border and never recover, whereas center-seeding is
      // self-correcting, and the row-to-row march below supplies continuity.
      const double seed = (W - 1) * 0.5;
      double best = 1e18;
      for (double cc : clusters) {
        const double d = std::abs(cc - seed);
        if (d < best) {best = d; picked = cc;}
      }
      if (best > max_jump_px) {continue;}
    } else {
      // Continuity: nearest cluster to where we are, within a max lateral jump.
      double best = 1e18;
      for (double cc : clusters) {
        const double d = std::abs(cc - current_col);
        if (d < best) {best = d; picked = cc;}
      }
      if (best > max_jump_px) {continue;}   // gap; keep current_col, re-acquire later
    }

    current_col = picked;
    double X, Y;
    ipm.bird_to_ground(picked, static_cast<double>(r), X, Y);
    res.points.push_back({X, Y});
    if (keep_debug) {
      res.bird_pts.emplace_back(static_cast<int>(std::lround(picked)), r);
    }
  }

  res.confidence = rows_with_clusters > 0
    ? std::min(1.0, static_cast<double>(res.points.size()) / rows_with_clusters)
    : 0.0;
  res.valid = static_cast<int>(res.points.size()) >= params_.min_points;

  if (res.valid) {
    // res.points[0] is the nearest (bottom) selected column.
    double c0, r0;
    ipm.ground_to_bird(res.points.front().x, res.points.front().y, c0, r0);
    prev_seed_col_ = have_prev_ ? (0.5 * prev_seed_col_ + 0.5 * c0) : c0;
    have_prev_ = true;
  }

  if (keep_debug) {res.mask = mask;}
  return res;
}

}  // namespace lane_pilot
