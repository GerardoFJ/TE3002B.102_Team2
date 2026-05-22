#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace traffic_light
{

enum class State : int
{
  Red = 0,
  Yellow = 1,
  Green = 2,
  None = 3,
};

const char * to_string(State s);

struct Detection
{
  double area = 0.0;
  std::vector<cv::Point> contour;
  bool has_contour = false;
};

class TrafficLightDetector
{
public:
  static constexpr int N_COLORS = 3;
  static constexpr std::array<int, N_COLORS> HUE_CENTERS = {0, 40, 65};
  static constexpr int MAX_HUE_DIST = 22;
  static constexpr int MIN_S = 65;
  static constexpr int MIN_V = 85;

  static constexpr int WHITE_V_MIN = 200;
  static constexpr int WHITE_S_MAX = 80;
  static constexpr int HALO_DILATE_PX = 7;
  static constexpr int HALO_OVERLAP_MIN = 20;

  static constexpr double MIN_AREA = 60.0;
  static constexpr double MIN_CIRCULARITY = 0.60;
  static constexpr double MIN_FILL_RATIO = 0.65;

  static constexpr int DEBUG_TILE_W = 320;
  static constexpr int DEBUG_TILE_H = 240;

  explicit TrafficLightDetector(bool debug = false);

  State detect_state(const cv::Mat & bgr, bool render_debug);

  const cv::Mat & last_debug_frame() const { return last_debug_frame_; }

private:
  void classify_pixels(const cv::Mat & hsv);
  void rescue_blown_cores(const cv::Mat & hsv);
  std::pair<double, std::vector<cv::Point>> best_circle(const cv::Mat & mask) const;
  bool is_roundish(const cv::Mat & mask) const;

  cv::Mat render_debug_frame(const cv::Mat & input) const;
  cv::Mat tile(
    const cv::Mat & img, const std::string & label,
    const cv::Scalar & label_color = cv::Scalar(255, 255, 255)) const;

  bool debug_;
  bool debug_window_ready_ = false;
  cv::Mat morph_kernel_;
  cv::Mat halo_kernel_;

  cv::Mat blurred_;
  cv::Mat hsv_;
  std::array<cv::Mat, N_COLORS> masks_;
  std::array<Detection, N_COLORS> detections_;

  cv::Mat white_mask_;
  cv::Mat rescued_mask_;
  cv::Mat labels_, stats_, centroids_;
  cv::Mat component_;
  cv::Mat dilated_;
  cv::Mat ring_;
  cv::Mat anded_;

  State state_ = State::None;
  cv::Mat last_debug_frame_;
};

}  // namespace traffic_light
