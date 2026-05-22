#include "traffic_light_detection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace traffic_light
{

namespace
{
const std::array<cv::Scalar, 4> COLOR_BGR = {{
  cv::Scalar(0, 0, 255),      // red
  cv::Scalar(0, 255, 255),    // yellow
  cv::Scalar(0, 255, 0),      // green
  cv::Scalar(200, 200, 200),  // none
}};

constexpr std::array<const char *, 3> COLOR_NAMES = {{"red", "yellow", "green"}};
}  // namespace

const char * to_string(State s)
{
  switch (s) {
    case State::Red: return "red";
    case State::Yellow: return "yellow";
    case State::Green: return "green";
    case State::None: return "none";
  }
  return "none";
}

TrafficLightDetector::TrafficLightDetector(bool debug)
: debug_(debug)
{
  morph_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  halo_kernel_ = cv::getStructuringElement(
    cv::MORPH_ELLIPSE,
    cv::Size(HALO_DILATE_PX * 2 + 1, HALO_DILATE_PX * 2 + 1));
}

State TrafficLightDetector::detect_state(const cv::Mat & bgr, bool render_debug)
{
  if (bgr.empty()) {
    state_ = State::None;
    return state_;
  }

  cv::GaussianBlur(bgr, blurred_, cv::Size(5, 5), 0);
  cv::cvtColor(blurred_, hsv_, cv::COLOR_BGR2HSV);

  classify_pixels(hsv_);
  rescue_blown_cores(hsv_);

  double best_area = 0.0;
  State best = State::None;
  for (int i = 0; i < N_COLORS; ++i) {
    auto [area, contour] = best_circle(masks_[i]);
    detections_[i].area = area;
    detections_[i].contour = std::move(contour);
    detections_[i].has_contour = !detections_[i].contour.empty();
    if (area > best_area) {
      best_area = area;
      best = static_cast<State>(i);
    }
  }
  state_ = best;

  if (render_debug || debug_) {
    last_debug_frame_ = render_debug_frame(bgr);
    if (debug_) {
      if (!debug_window_ready_) {
        cv::namedWindow("Traffic Light Debug", cv::WINDOW_NORMAL);
        cv::resizeWindow(
          "Traffic Light Debug", DEBUG_TILE_W * 3, DEBUG_TILE_H * 2);
        debug_window_ready_ = true;
      }
      cv::imshow("Traffic Light Debug", last_debug_frame_);
      cv::waitKey(1);
    }
  } else {
    last_debug_frame_.release();
  }

  return state_;
}

void TrafficLightDetector::classify_pixels(const cv::Mat & hsv)
{
  const int rows = hsv.rows;
  const int cols = hsv.cols;

  for (int i = 0; i < N_COLORS; ++i) {
    if (masks_[i].rows != rows || masks_[i].cols != cols ||
      masks_[i].type() != CV_8UC1)
    {
      masks_[i].create(rows, cols, CV_8UC1);
    }
    masks_[i].setTo(0);
  }

  // Single-pass nearest-hue-center classification. Avoids the per-channel
  // split + distance stack + argmin + take_along_axis chain the Python
  // version did (which allocated ~6 image-sized buffers per frame).
  for (int y = 0; y < rows; ++y) {
    const cv::Vec3b * hsv_row = hsv.ptr<cv::Vec3b>(y);
    uint8_t * mr0 = masks_[0].ptr<uint8_t>(y);
    uint8_t * mr1 = masks_[1].ptr<uint8_t>(y);
    uint8_t * mr2 = masks_[2].ptr<uint8_t>(y);

    for (int x = 0; x < cols; ++x) {
      const cv::Vec3b & p = hsv_row[x];
      const int s = p[1];
      const int v = p[2];
      if (s < MIN_S || v < MIN_V) {
        continue;
      }
      const int h = p[0];

      int best_idx = 0;
      int best_dist = 1000;
      for (int c = 0; c < N_COLORS; ++c) {
        int d = std::abs(h - HUE_CENTERS[c]);
        if (d > 90) {
          d = 180 - d;
        }
        if (d < best_dist) {
          best_dist = d;
          best_idx = c;
        }
      }
      if (best_dist > MAX_HUE_DIST) {
        continue;
      }
      switch (best_idx) {
        case 0: mr0[x] = 255; break;
        case 1: mr1[x] = 255; break;
        case 2: mr2[x] = 255; break;
      }
    }
  }

  for (int i = 0; i < N_COLORS; ++i) {
    cv::morphologyEx(masks_[i], masks_[i], cv::MORPH_OPEN, morph_kernel_);
    cv::morphologyEx(masks_[i], masks_[i], cv::MORPH_CLOSE, morph_kernel_);
  }
}

void TrafficLightDetector::rescue_blown_cores(const cv::Mat & hsv)
{
  const int rows = hsv.rows;
  const int cols = hsv.cols;

  if (white_mask_.rows != rows || white_mask_.cols != cols ||
    white_mask_.type() != CV_8UC1)
  {
    white_mask_.create(rows, cols, CV_8UC1);
  }
  white_mask_.setTo(0);

  for (int y = 0; y < rows; ++y) {
    const cv::Vec3b * row = hsv.ptr<cv::Vec3b>(y);
    uint8_t * out = white_mask_.ptr<uint8_t>(y);
    for (int x = 0; x < cols; ++x) {
      const cv::Vec3b & p = row[x];
      if (p[2] >= WHITE_V_MIN && p[1] <= WHITE_S_MAX) {
        out[x] = 255;
      }
    }
  }
  cv::morphologyEx(white_mask_, white_mask_, cv::MORPH_OPEN, morph_kernel_);

  if (rescued_mask_.rows != rows || rescued_mask_.cols != cols ||
    rescued_mask_.type() != CV_8UC1)
  {
    rescued_mask_.create(rows, cols, CV_8UC1);
  }
  rescued_mask_.setTo(0);

  if (cv::countNonZero(white_mask_) == 0) {
    return;
  }

  const int num_labels = cv::connectedComponentsWithStats(
    white_mask_, labels_, stats_, centroids_, 8);

  for (int label = 1; label < num_labels; ++label) {
    const int area = stats_.at<int>(label, cv::CC_STAT_AREA);
    if (area < static_cast<int>(MIN_AREA)) {
      continue;
    }

    if (component_.rows != rows || component_.cols != cols ||
      component_.type() != CV_8UC1)
    {
      component_.create(rows, cols, CV_8UC1);
    }
    component_.setTo(0);

    const int bx = stats_.at<int>(label, cv::CC_STAT_LEFT);
    const int by = stats_.at<int>(label, cv::CC_STAT_TOP);
    const int bw = stats_.at<int>(label, cv::CC_STAT_WIDTH);
    const int bh = stats_.at<int>(label, cv::CC_STAT_HEIGHT);
    for (int y = by; y < by + bh; ++y) {
      const int32_t * lab_row = labels_.ptr<int32_t>(y);
      uint8_t * comp_row = component_.ptr<uint8_t>(y);
      for (int x = bx; x < bx + bw; ++x) {
        if (lab_row[x] == label) {
          comp_row[x] = 255;
        }
      }
    }

    if (!is_roundish(component_)) {
      continue;
    }

    cv::dilate(component_, dilated_, halo_kernel_);
    cv::subtract(dilated_, component_, ring_);

    int best_overlap = 0;
    int best_color = -1;
    for (int c = 0; c < N_COLORS; ++c) {
      cv::bitwise_and(ring_, masks_[c], anded_);
      const int n = cv::countNonZero(anded_);
      if (n > best_overlap) {
        best_overlap = n;
        best_color = c;
      }
    }
    if (best_color < 0 || best_overlap < HALO_OVERLAP_MIN) {
      continue;
    }

    cv::bitwise_or(masks_[best_color], component_, masks_[best_color]);
    cv::bitwise_or(rescued_mask_, component_, rescued_mask_);
  }
}

bool TrafficLightDetector::is_roundish(const cv::Mat & mask) const
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return false;
  }
  size_t best_idx = 0;
  double best_area = 0.0;
  for (size_t i = 0; i < contours.size(); ++i) {
    const double a = cv::contourArea(contours[i]);
    if (a > best_area) {
      best_area = a;
      best_idx = i;
    }
  }
  if (best_area <= 0.0) {
    return false;
  }
  const double perimeter = cv::arcLength(contours[best_idx], true);
  if (perimeter <= 0.0) {
    return false;
  }
  const double circularity = 4.0 * CV_PI * best_area / (perimeter * perimeter);
  return circularity >= MIN_CIRCULARITY;
}

std::pair<double, std::vector<cv::Point>>
TrafficLightDetector::best_circle(const cv::Mat & mask) const
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double best_area = 0.0;
  std::vector<cv::Point> best;
  for (auto & c : contours) {
    const double area = cv::contourArea(c);
    if (area < MIN_AREA) {
      continue;
    }
    const double perimeter = cv::arcLength(c, true);
    if (perimeter <= 0.0) {
      continue;
    }
    const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    if (circularity < MIN_CIRCULARITY) {
      continue;
    }
    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(c, center, radius);
    if (radius <= 0.0f) {
      continue;
    }
    const double fill = area / (CV_PI * radius * radius);
    if (fill < MIN_FILL_RATIO) {
      continue;
    }
    if (area > best_area) {
      best_area = area;
      best = c;
    }
  }
  return {best_area, std::move(best)};
}

cv::Mat TrafficLightDetector::tile(
  const cv::Mat & img, const std::string & label,
  const cv::Scalar & label_color) const
{
  cv::Mat canvas = cv::Mat::zeros(DEBUG_TILE_H, DEBUG_TILE_W, CV_8UC3);
  if (!img.empty() && img.rows >= 2 && img.cols >= 2) {
    cv::Mat src = img;
    if (src.channels() == 1) {
      cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
    }
    const double scale = std::min(
      static_cast<double>(DEBUG_TILE_W) / src.cols,
      static_cast<double>(DEBUG_TILE_H) / src.rows);
    const int new_w = std::max(1, static_cast<int>(std::lround(src.cols * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(src.rows * scale)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
    const int x_off = (DEBUG_TILE_W - new_w) / 2;
    const int y_off = (DEBUG_TILE_H - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x_off, y_off, new_w, new_h)));
  }
  cv::putText(
    canvas, label, cv::Point(6, 18),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
  cv::putText(
    canvas, label, cv::Point(6, 18),
    cv::FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv::LINE_AA);
  return canvas;
}

cv::Mat TrafficLightDetector::render_debug_frame(const cv::Mat & input) const
{
  const cv::Scalar state_color = COLOR_BGR[static_cast<int>(state_)];
  const char * state_name = to_string(state_);

  // Tile 1: camera + accepted circles
  cv::Mat cam_overlay = input.clone();
  for (int c = 0; c < N_COLORS; ++c) {
    if (!detections_[c].has_contour) {
      continue;
    }
    cv::Point2f center;
    float radius = 0.0f;
    cv::minEnclosingCircle(detections_[c].contour, center, radius);
    cv::circle(
      cam_overlay,
      cv::Point(cvRound(center.x), cvRound(center.y)),
      cvRound(radius), COLOR_BGR[c], 2);
  }
  char buf[160];
  std::snprintf(buf, sizeof(buf), "1. camera  state=%s", state_name);
  cv::Mat cam_tile = tile(cam_overlay, buf, state_color);

  // Tile 2: white-out rescue
  cv::Mat white_tile;
  if (!white_mask_.empty()) {
    cv::Mat white_viz;
    cv::cvtColor(white_mask_, white_viz, cv::COLOR_GRAY2BGR);
    white_viz /= 2;
    if (!rescued_mask_.empty()) {
      white_viz.setTo(cv::Scalar(255, 255, 255), rescued_mask_);
    }
    white_tile = tile(white_viz, "2. white-out rescue");
  } else {
    white_tile = tile(cv::Mat(), "2. white-out rescue");
  }

  // Tile 3: winner crop
  cv::Mat winner_tile;
  if (state_ != State::None) {
    const int idx = static_cast<int>(state_);
    cv::Mat isolated;
    cv::bitwise_and(input, input, isolated, masks_[idx]);
    if (detections_[idx].has_contour) {
      std::vector<std::vector<cv::Point>> cs = {detections_[idx].contour};
      cv::drawContours(isolated, cs, -1, cv::Scalar(255, 255, 255), 2);
    }
    std::snprintf(
      buf, sizeof(buf), "3. winner: %s  area=%d",
      state_name, static_cast<int>(detections_[idx].area));
    winner_tile = tile(isolated, buf, state_color);
  } else {
    winner_tile = tile(cv::Mat(), "3. no detection");
  }

  cv::Mat row1;
  cv::hconcat(std::vector<cv::Mat>{cam_tile, white_tile, winner_tile}, row1);

  std::vector<cv::Mat> panels;
  panels.reserve(N_COLORS);
  for (int c = 0; c < N_COLORS; ++c) {
    cv::Mat viz = cv::Mat::zeros(masks_[c].size(), CV_8UC3);
    viz.setTo(COLOR_BGR[c], masks_[c]);
    if (detections_[c].has_contour) {
      std::vector<std::vector<cv::Point>> cs = {detections_[c].contour};
      cv::drawContours(viz, cs, -1, cv::Scalar(255, 255, 255), 2);
    }
    std::snprintf(
      buf, sizeof(buf), "%d. %s mask  area=%d",
      c + 4, COLOR_NAMES[c], static_cast<int>(detections_[c].area));
    panels.push_back(tile(viz, buf, COLOR_BGR[c]));
  }
  cv::Mat row2;
  cv::hconcat(panels, row2);

  cv::Mat out;
  cv::vconcat(std::vector<cv::Mat>{row1, row2}, out);
  return out;
}

}  // namespace traffic_light
