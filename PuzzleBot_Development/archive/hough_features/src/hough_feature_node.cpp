// hough_feature_node — Phase 1 of the ML-based line follower.
//
// Subscribes to the camera's JPEG topic, runs the HoughDetector, and
// publishes a fixed-order 14-dim feature vector on /hough_features. An
// optional annotated debug image is published lazily — encoded only when
// the user has explicitly enabled it AND someone is subscribed.
//
// No /cmd_vel is published. That's Phase 3.
//
// Decoding patterns are copied verbatim from traffic_light/src/detect.cpp
// (cv::imdecode, SensorDataQoS, lazy JPEG-encoded debug). The hardware
// nvjpegdec path is NOT used: see the comment at the top of
// traffic_light/src/detect.cpp explaining why it segfaults when loaded
// into a process that already has OpenCV+CUDA linked.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/multi_array_dimension.hpp"

#include "hough_detector.hpp"

namespace
{

sensor_msgs::msg::CompressedImage bgr_to_compressed_jpeg(
  const cv::Mat & bgr,
  const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id,
  int quality)
{
  sensor_msgs::msg::CompressedImage msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.format = "jpeg";
  const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
  cv::imencode(".jpg", bgr, msg.data, params);
  return msg;
}

}  // namespace

class HoughFeatureNode : public rclcpp::Node
{
public:
  HoughFeatureNode()
  : Node("hough_feature_node")
  {
    const std::string input_topic = declare_parameter<std::string>(
      "input_topic", "/camera/image_rect/compressed");
    const std::string output_topic = declare_parameter<std::string>(
      "output_topic", "/hough_features");

    hough_features::DetectorParams p;
    p.canny1 = declare_parameter<double>("canny1", 60.0);
    p.canny2 = declare_parameter<double>("canny2", 160.0);
    p.hough_threshold = static_cast<int>(
      declare_parameter<int>("hough_threshold", 35));
    p.min_line_length = declare_parameter<double>("min_line_length", 25.0);
    p.max_line_gap = declare_parameter<double>("max_line_gap", 15.0);
    p.roi_top_frac = std::clamp(
      declare_parameter<double>("roi_top_frac", 0.40), 0.0, 0.99);
    p.roi_bottom_frac = std::clamp(
      declare_parameter<double>("roi_bottom_frac", 1.00), p.roi_top_frac + 0.01, 1.0);
    p.roi_left_frac = std::clamp(
      declare_parameter<double>("roi_left_frac", 0.00), 0.0, 0.49);
    p.roi_right_frac = std::clamp(
      declare_parameter<double>("roi_right_frac", 1.00), 0.51, 1.0);
    p.detect_scale = std::clamp(
      declare_parameter<double>("detect_scale", 0.5), 0.1, 1.0);
    p.max_vp_pairs = static_cast<int>(
      declare_parameter<int>("max_vp_pairs", 80));
    detector_ = std::make_unique<hough_features::HoughDetector>(p);

    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_jpeg_quality_ = std::clamp(
      static_cast<int>(declare_parameter<int>("debug_jpeg_quality", 70)),
      1, 100);

    // OpenCV thread pool fights for cores on the Nano; the per-frame work
    // is small enough that single-threaded SIMD wins.
    cv::setNumThreads(1);

    features_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      output_topic, rclcpp::QoS(10));

    if (publish_debug_image_) {
      debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        output_topic + "/debug_image/compressed", 5);
    }

    image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      input_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&HoughFeatureNode::onImage, this, std::placeholders::_1));

    last_stats_time_ = now();
    stats_timer_ = create_wall_timer(
      std::chrono::seconds(5),
      std::bind(&HoughFeatureNode::logStats, this));

    RCLCPP_INFO(
      get_logger(),
      "hough_feature_node started "
      "(input=%s out=%s scale=%.2f canny=%.0f/%.0f "
      "hough_thr=%d min_len=%.0f max_gap=%.0f).",
      input_topic.c_str(), output_topic.c_str(),
      p.detect_scale, p.canny1, p.canny2,
      p.hough_threshold, p.min_line_length, p.max_line_gap);
  }

private:
  void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
  {
    if (msg->data.empty()) {
      return;
    }
    cv::Mat bgr = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      return;
    }

    const auto t_start = std::chrono::steady_clock::now();
    const bool want_debug =
      publish_debug_image_ && debug_pub_ &&
      debug_pub_->get_subscription_count() > 0;

    auto result = detector_->detect(bgr, want_debug);

    const auto t_end = std::chrono::steady_clock::now();
    const double detect_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();
    detect_ms_sum_ += detect_ms;
    ++frame_count_;

    publishFeatures(result.features, msg->header.stamp);

    if (want_debug) {
      publishDebug(bgr, result, msg->header.stamp);
    }
  }

  void publishFeatures(
    const std::array<float, hough_features::N_FEATURES> & features,
    const builtin_interfaces::msg::Time & /*stamp*/)
  {
    std_msgs::msg::Float32MultiArray msg;
    // Always populate the layout so Python consumers can assert shape.
    msg.layout.data_offset = 0;
    std_msgs::msg::MultiArrayDimension dim;
    dim.label = "features";
    dim.size = hough_features::N_FEATURES;
    dim.stride = hough_features::N_FEATURES;
    msg.layout.dim.push_back(dim);
    msg.data.assign(features.begin(), features.end());
    features_pub_->publish(msg);
  }

  void publishDebug(
    const cv::Mat & original_bgr,
    const hough_features::DetectionResult & result,
    const builtin_interfaces::msg::Time & stamp)
  {
    // Draw onto a downscaled-to-processing-frame copy so coordinates
    // match the detector's frame directly. Reverse the resize at the end
    // to keep the published image readable.
    const double scale = detector_->params().detect_scale;
    cv::Mat scaled;
    if (scale < 0.999) {
      cv::resize(
        original_bgr, scaled, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
      scaled = original_bgr.clone();
    }

    // ROI rectangle.
    const cv::Rect roi = result.roi_in_original;
    cv::rectangle(
      scaled, roi, cv::Scalar(50, 200, 50), 1);

    // Hough segments (in ROI coords) → shift back into the scaled frame.
    for (const auto & s : result.segments) {
      const cv::Point p1(
        static_cast<int>(std::round(s[0])) + roi.x,
        static_cast<int>(std::round(s[1])) + roi.y);
      const cv::Point p2(
        static_cast<int>(std::round(s[2])) + roi.x,
        static_cast<int>(std::round(s[3])) + roi.y);
      cv::line(scaled, p1, p2, cv::Scalar(0, 0, 255), 2);
    }

    // center_bottom (green dot at the bottom of the ROI).
    const float center_bottom_roi = result.features[hough_features::CENTER_BOTTOM];
    const int cb_x = static_cast<int>(std::round(center_bottom_roi)) + roi.x;
    const int cb_y = roi.y + roi.height - 4;
    cv::circle(scaled, cv::Point(cb_x, cb_y), 6, cv::Scalar(0, 220, 0), -1);
    cv::line(
      scaled, cv::Point(roi.x + roi.width / 2, cb_y),
      cv::Point(cb_x, cb_y), cv::Scalar(0, 220, 0), 1);

    // vanishing_x (cyan dot near horizon i.e. top of ROI).
    const float vx_roi = result.features[hough_features::VANISHING_X];
    const int vx_x = static_cast<int>(std::round(vx_roi)) + roi.x;
    const int vx_y = roi.y + 8;
    cv::circle(scaled, cv::Point(vx_x, vx_y), 6, cv::Scalar(255, 255, 0), -1);

    // Image-center vertical reference.
    cv::line(
      scaled, cv::Point(scaled.cols / 2, 0),
      cv::Point(scaled.cols / 2, scaled.rows - 1),
      cv::Scalar(120, 120, 120), 1);

    // HUD line.
    char buf[160];
    std::snprintf(
      buf, sizeof(buf),
      "n=%d cerr=%+.2f verr=%+.2f bal=%+.2f conf=%.2f",
      static_cast<int>(result.features[hough_features::N_LINES]),
      result.features[hough_features::CENTER_ERROR],
      result.features[hough_features::VANISHING_ERROR],
      result.features[hough_features::BALANCE_LR],
      result.features[hough_features::CONFIDENCE]);
    cv::putText(
      scaled, buf, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(
      scaled, buf, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    debug_pub_->publish(
      bgr_to_compressed_jpeg(
        scaled, stamp, "hough_features", debug_jpeg_quality_));
  }

  void logStats()
  {
    const auto now_t = now();
    const double dt = (now_t - last_stats_time_).seconds();
    if (dt > 0.0 && frame_count_ > 0) {
      const double rate = frame_count_ / dt;
      const double mean_ms = detect_ms_sum_ / frame_count_;
      RCLCPP_INFO(
        get_logger(),
        "rate=%.1f fps  mean_detect=%.2f ms  frames=%lu",
        rate, mean_ms, static_cast<unsigned long>(frame_count_));
    }
    frame_count_ = 0;
    detect_ms_sum_ = 0.0;
    last_stats_time_ = now_t;
  }

  std::unique_ptr<hough_features::HoughDetector> detector_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr features_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  bool publish_debug_image_ = true;
  int debug_jpeg_quality_ = 70;

  uint64_t frame_count_ = 0;
  double detect_ms_sum_ = 0.0;
  rclcpp::Time last_stats_time_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HoughFeatureNode>());
  rclcpp::shutdown();
  return 0;
}
