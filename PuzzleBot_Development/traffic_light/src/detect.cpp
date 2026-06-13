#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

#include "traffic_light_detection.hpp"

namespace
{

cv::Mat imgmsg_to_bgr(const sensor_msgs::msg::Image & msg)
{
  const int rows = static_cast<int>(msg.height);
  const int cols = static_cast<int>(msg.width);
  if (rows <= 0 || cols <= 0 || msg.data.empty()) {
    return cv::Mat();
  }
  uint8_t * data = const_cast<uint8_t *>(msg.data.data());

  if (msg.encoding == "bgr8") {
    return cv::Mat(rows, cols, CV_8UC3, data).clone();
  }
  if (msg.encoding == "rgb8") {
    cv::Mat src(rows, cols, CV_8UC3, data);
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  if (msg.encoding == "mono8") {
    cv::Mat gray(rows, cols, CV_8UC1, data);
    cv::Mat bgr;
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  }
  // Fallback: assume bgr8.
  return cv::Mat(rows, cols, CV_8UC3, data).clone();
}

sensor_msgs::msg::Image bgr_to_imgmsg(
  const cv::Mat & bgr, const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id)
{
  sensor_msgs::msg::Image msg;
  msg.height = static_cast<uint32_t>(bgr.rows);
  msg.width = static_cast<uint32_t>(bgr.cols);
  msg.encoding = "bgr8";
  msg.is_bigendian = 0;
  msg.step = static_cast<uint32_t>(bgr.cols * 3);
  msg.data.resize(static_cast<size_t>(msg.step) * msg.height);
  if (bgr.isContinuous()) {
    std::memcpy(msg.data.data(), bgr.data, msg.data.size());
  } else {
    for (int r = 0; r < bgr.rows; ++r) {
      std::memcpy(
        msg.data.data() + static_cast<size_t>(r) * msg.step,
        bgr.ptr(r), msg.step);
    }
  }
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  return msg;
}

}  // namespace

class TrafficLightNode : public rclcpp::Node
{
public:
  TrafficLightNode()
  : Node("traffic_light_detector")
  {
    debug_ = this->declare_parameter<bool>("debug", false);
    confirm_frames_ = std::max(
      1, static_cast<int>(this->declare_parameter<int>("confirm_frames", 3)));
    const double go_hz = std::max(
      0.1, this->declare_parameter<double>("go_publish_hz", 5.0));

    detector_ = std::make_unique<traffic_light::TrafficLightDetector>(debug_);

    state_pub_ = this->create_publisher<std_msgs::msg::String>(
      "/traffic_light/state", 10);
    go_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/traffic_light/go", 10);
    debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/traffic_light/debug_image", 5);

    // KeepLast(1): always process the freshest camera frame and drop stale
    // ones instead of working through a 10-frame backlog (which made
    // detection lag noticeable on slower hardware).
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/video_source/raw", rclcpp::QoS(1),
      std::bind(&TrafficLightNode::image_callback, this,
        std::placeholders::_1));

    go_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / go_hz)),
      std::bind(&TrafficLightNode::publish_go, this));

    RCLCPP_INFO(
      this->get_logger(),
      "Traffic light detector started (confirm_frames=%d, go_publish_hz=%g).",
      confirm_frames_, go_hz);
  }

private:
  void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv::Mat frame = imgmsg_to_bgr(*msg);
    if (frame.empty()) {
      return;
    }

    // Skip building the debug composite when nobody's watching — debug
    // rendering (clone + 6 tiles + hconcat/vconcat + putText) was a big
    // chunk of the per-frame cost in the Python version.
    const bool want_debug =
      debug_ || debug_image_pub_->get_subscription_count() > 0;

    const traffic_light::State raw =
      detector_->detect_state(frame, want_debug);
    const char * raw_name = traffic_light::to_string(raw);

    std_msgs::msg::String state_msg;
    state_msg.data = raw_name;
    state_pub_->publish(state_msg);

    if (want_debug && !detector_->last_debug_frame().empty()) {
      debug_image_pub_->publish(
        bgr_to_imgmsg(
          detector_->last_debug_frame(), msg->header.stamp,
          "traffic_light_debug"));
    }

    const auto confirmed = debounce(raw);
    if (confirmed.has_value()) {
      on_confirmed_state(*confirmed);
    }
  }

  std::optional<traffic_light::State> debounce(traffic_light::State raw)
  {
    if (have_candidate_ && raw == candidate_state_) {
      ++candidate_count_;
    } else {
      candidate_state_ = raw;
      candidate_count_ = 1;
      have_candidate_ = true;
    }
    if (candidate_count_ >= confirm_frames_ &&
      (!have_confirmed_ || raw != confirmed_state_))
    {
      confirmed_state_ = raw;
      have_confirmed_ = true;
      return raw;
    }
    return std::nullopt;
  }

  void on_confirmed_state(traffic_light::State s)
  {
    const bool go =
      s == traffic_light::State::Green || s == traffic_light::State::None;
    current_go_ = go;
    RCLCPP_INFO(
      this->get_logger(), "detected: %s  -> go=%s",
      traffic_light::to_string(s), go ? "true" : "false");
  }

  void publish_go()
  {
    std_msgs::msg::Bool b;
    b.data = current_go_;
    go_pub_->publish(b);
  }

  bool debug_;
  int confirm_frames_;
  std::unique_ptr<traffic_light::TrafficLightDetector> detector_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr go_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::TimerBase::SharedPtr go_timer_;

  traffic_light::State candidate_state_ = traffic_light::State::None;
  int candidate_count_ = 0;
  bool have_candidate_ = false;
  traffic_light::State confirmed_state_ = traffic_light::State::None;
  bool have_confirmed_ = false;
  bool current_go_ = true;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrafficLightNode>());
  rclcpp::shutdown();
  return 0;
}
