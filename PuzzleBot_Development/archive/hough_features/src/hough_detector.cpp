#include "hough_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace hough_features
{

namespace
{

constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfPi = kPi * 0.5f;

// Collapse a raw segment angle from atan2's [-pi, pi] range into the
// orientation range [-pi/2, pi/2]. Segments don't have a canonical
// direction — a horizontal segment can come back as ~0 or ~+/-pi
// depending on endpoint ordering, which would make mean_angle/std_angle
// bimodal and useless to a downstream regressor. Treating it as an
// orientation (not a direction) fixes that.
//
// This is the one deliberate divergence from the PDF Python code.
inline float collapse_to_orientation(float a)
{
  if (a > kHalfPi) {
    a -= kPi;
  } else if (a < -kHalfPi) {
    a += kPi;
  }
  return a;
}

}  // namespace

HoughDetector::HoughDetector(DetectorParams params)
: params_(std::move(params))
{
}

float HoughDetector::line_x_at_y(
  float x1, float y1, float x2, float y2, float y_query)
{
  if (std::abs(y2 - y1) < 1e-6f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float t = (y_query - y1) / (y2 - y1);
  return x1 + t * (x2 - x1);
}

float HoughDetector::estimate_corridor_center_at_bottom(
  const std::vector<cv::Vec4f> & segments, int width, int height)
{
  const float w_f = static_cast<float>(width);
  const float half = w_f * 0.5f;

  if (segments.empty()) {
    return half;
  }

  const float bottom_y = static_cast<float>(height - 1);

  // Collect every segment's intersection with the bottom row, dropping
  // wild values that imply the segment was effectively horizontal or
  // pointing far outside the frame.
  std::vector<float> xs_bottom;
  xs_bottom.reserve(segments.size());
  for (const auto & s : segments) {
    const float xb = line_x_at_y(s[0], s[1], s[2], s[3], bottom_y);
    if (std::isfinite(xb) && xb >= -0.5f * w_f && xb <= 1.5f * w_f) {
      xs_bottom.push_back(xb);
    }
  }

  if (xs_bottom.size() < 2) {
    return half;
  }

  // Partition into "left of center" and "right of center" candidates.
  // When both sides have intersections, the corridor center is best
  // approximated as the midpoint of the two medians (PDF Section 5).
  std::vector<float> left;
  std::vector<float> right;
  left.reserve(xs_bottom.size());
  right.reserve(xs_bottom.size());
  for (const float x : xs_bottom) {
    if (x < half) {
      left.push_back(x);
    } else {
      right.push_back(x);
    }
  }

  auto median = [](std::vector<float> & v) {
    const size_t n = v.size();
    auto mid = v.begin() + n / 2;
    std::nth_element(v.begin(), mid, v.end());
    if (n % 2 == 1) {
      return *mid;
    }
    // Even count: average the two middle elements.
    const float upper = *mid;
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    const float lower = *(v.begin() + n / 2 - 1);
    return 0.5f * (lower + upper);
  };

  if (!left.empty() && !right.empty()) {
    return 0.5f * (median(left) + median(right));
  }
  return median(xs_bottom);
}

float HoughDetector::estimate_vanishing_x(
  const std::vector<cv::Vec4f> & segments, int width, int height,
  int max_pairs)
{
  const float w_f = static_cast<float>(width);
  const float h_f = static_cast<float>(height);
  const float half = w_f * 0.5f;

  if (segments.size() < 2) {
    return half;
  }

  std::vector<float> intersections;
  intersections.reserve(static_cast<size_t>(max_pairs));

  // For each pair of segments, compute the intersection point. PDF
  // Section 5 caps the total pair count to bound the O(N^2) cost.
  int count = 0;
  for (size_t i = 0; i < segments.size() && count < max_pairs; ++i) {
    const float x1 = segments[i][0];
    const float y1 = segments[i][1];
    const float x2 = segments[i][2];
    const float y2 = segments[i][3];
    const float a1 = y2 - y1;
    const float b1 = x1 - x2;
    const float c1 = a1 * x1 + b1 * y1;

    for (size_t j = i + 1; j < segments.size() && count < max_pairs; ++j) {
      const float x3 = segments[j][0];
      const float y3 = segments[j][1];
      const float x4 = segments[j][2];
      const float y4 = segments[j][3];
      const float a2 = y4 - y3;
      const float b2 = x3 - x4;
      const float c2 = a2 * x3 + b2 * y3;

      const float det = a1 * b2 - a2 * b1;
      if (std::abs(det) < 1e-6f) {
        // Near-parallel segments — intersection is at infinity, skip.
        continue;
      }

      const float ix = (b2 * c1 - b1 * c2) / det;
      const float iy = (a1 * c2 - a2 * c1) / det;

      // Keep only intersections within a generous box around the frame
      // (PDF clamp). This rejects noise pairs whose intersection point
      // is far outside the camera FOV.
      if (ix >= -w_f && ix <= 2.0f * w_f && iy >= -h_f && iy <= h_f) {
        intersections.push_back(ix);
        ++count;
      }
    }
  }

  if (intersections.empty()) {
    return half;
  }

  auto mid = intersections.begin() + intersections.size() / 2;
  std::nth_element(intersections.begin(), mid, intersections.end());
  return *mid;
}

std::array<float, N_FEATURES> HoughDetector::extract_hough_features(
  const std::vector<cv::Vec4f> & segments, int width, int height,
  int max_vp_pairs)
{
  std::array<float, N_FEATURES> f{};
  const float w_f = static_cast<float>(width);
  const float half = w_f * 0.5f;

  // Zero-segment short-circuit. Matches the PDF "no Hough lines" path.
  if (segments.empty()) {
    f[N_LINES] = 0.0f;
    f[MEAN_ANGLE] = 0.0f;
    f[STD_ANGLE] = 0.0f;
    f[MEAN_ABS_ANGLE] = 0.0f;
    f[MEAN_LENGTH] = 0.0f;
    f[MAX_LENGTH] = 0.0f;
    f[LEFT_COUNT] = 0.0f;
    f[RIGHT_COUNT] = 0.0f;
    f[BALANCE_LR] = 0.0f;
    f[CENTER_BOTTOM] = half;
    f[CENTER_ERROR] = 0.0f;
    f[VANISHING_X] = half;
    f[VANISHING_ERROR] = 0.0f;
    f[CONFIDENCE] = 0.0f;
    return f;
  }

  const size_t n = segments.size();
  std::vector<float> lengths(n);
  std::vector<float> angles(n);
  std::vector<float> abs_angles(n);
  std::vector<float> midxs(n);

  for (size_t i = 0; i < n; ++i) {
    const float x1 = segments[i][0];
    const float y1 = segments[i][1];
    const float x2 = segments[i][2];
    const float y2 = segments[i][3];
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    lengths[i] = std::sqrt(dx * dx + dy * dy);
    // Collapse to orientation [-pi/2, pi/2] — see comment in
    // collapse_to_orientation() for why this differs from raw atan2.
    angles[i] = collapse_to_orientation(std::atan2(dy, dx));
    abs_angles[i] = std::abs(angles[i]);
    midxs[i] = 0.5f * (x1 + x2);
  }

  const float mean_angle =
    std::accumulate(angles.begin(), angles.end(), 0.0f) / n;
  float var = 0.0f;
  if (n > 1) {
    for (const float a : angles) {
      const float d = a - mean_angle;
      var += d * d;
    }
    var /= static_cast<float>(n);  // numpy.std uses biased estimator (ddof=0)
  }
  const float std_angle = std::sqrt(var);
  const float mean_abs_angle =
    std::accumulate(abs_angles.begin(), abs_angles.end(), 0.0f) / n;
  const float mean_length =
    std::accumulate(lengths.begin(), lengths.end(), 0.0f) / n;
  const float max_length = *std::max_element(lengths.begin(), lengths.end());

  int left_count = 0;
  int right_count = 0;
  for (const float mx : midxs) {
    if (mx < half) {
      ++left_count;
    } else {
      ++right_count;
    }
  }
  const float balance_lr =
    static_cast<float>(right_count - left_count) /
    static_cast<float>(std::max(1, static_cast<int>(n)));

  const float center_bottom =
    estimate_corridor_center_at_bottom(segments, width, height);
  const float center_error = (center_bottom - half) / half;

  const float vanishing_x =
    estimate_vanishing_x(segments, width, height, max_vp_pairs);
  const float vanishing_error = (vanishing_x - half) / half;

  // PDF confidence heuristic: more segments AND longer mean length both
  // raise confidence, capped at 1.
  const float confidence =
    std::min(1.0f, static_cast<float>(n) / 12.0f) *
    std::clamp(mean_length / 80.0f, 0.0f, 1.0f);

  f[N_LINES] = static_cast<float>(n);
  f[MEAN_ANGLE] = mean_angle;
  f[STD_ANGLE] = std_angle;
  f[MEAN_ABS_ANGLE] = mean_abs_angle;
  f[MEAN_LENGTH] = mean_length;
  f[MAX_LENGTH] = max_length;
  f[LEFT_COUNT] = static_cast<float>(left_count);
  f[RIGHT_COUNT] = static_cast<float>(right_count);
  f[BALANCE_LR] = balance_lr;
  f[CENTER_BOTTOM] = center_bottom;
  f[CENTER_ERROR] = center_error;
  f[VANISHING_X] = vanishing_x;
  f[VANISHING_ERROR] = vanishing_error;
  f[CONFIDENCE] = confidence;
  return f;
}

DetectionResult HoughDetector::detect(
  const cv::Mat & bgr, bool keep_edges_for_debug)
{
  DetectionResult result;
  if (bgr.empty()) {
    result.features = extract_hough_features({}, 1, 1, params_.max_vp_pairs);
    return result;
  }

  // Apply downscale before ROI cropping — keeps ROI fractions in the
  // same domain whether the user resized first or not.
  cv::Mat scaled;
  if (params_.detect_scale < 0.999) {
    cv::resize(
      bgr, scaled, cv::Size(), params_.detect_scale, params_.detect_scale,
      cv::INTER_AREA);
  } else {
    scaled = bgr;
  }

  const int W = scaled.cols;
  const int H = scaled.rows;
  const int x0 = std::clamp(
    static_cast<int>(std::round(params_.roi_left_frac * W)), 0, W - 2);
  const int x1 = std::clamp(
    static_cast<int>(std::round(params_.roi_right_frac * W)), x0 + 1, W);
  const int y0 = std::clamp(
    static_cast<int>(std::round(params_.roi_top_frac * H)), 0, H - 2);
  const int y1 = std::clamp(
    static_cast<int>(std::round(params_.roi_bottom_frac * H)), y0 + 1, H);
  const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
  result.roi_in_original = roi;

  cv::Mat roi_img = scaled(roi);
  cv::Mat gray;
  cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat edges;
  cv::Canny(gray, edges, params_.canny1, params_.canny2);

  std::vector<cv::Vec4i> lines_i;
  cv::HoughLinesP(
    edges, lines_i,
    /*rho=*/1.0,
    /*theta=*/CV_PI / 180.0,
    params_.hough_threshold,
    params_.min_line_length,
    params_.max_line_gap);

  // Convert to float vectors so the geometry math doesn't keep
  // round-tripping through ints.
  result.segments.reserve(lines_i.size());
  for (const auto & l : lines_i) {
    result.segments.emplace_back(
      static_cast<float>(l[0]), static_cast<float>(l[1]),
      static_cast<float>(l[2]), static_cast<float>(l[3]));
  }

  result.processed_width = roi.width;
  result.processed_height = roi.height;
  result.features = extract_hough_features(
    result.segments, roi.width, roi.height, params_.max_vp_pairs);

  if (keep_edges_for_debug) {
    result.edges = std::move(edges);
  }
  return result;
}

}  // namespace hough_features
