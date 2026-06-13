// Line follower for the Puzzlebot.
//
// Detects up to three black lines in the lower portion of the camera image
// and steers the robot toward the center one. Designed to be robust against
// brief gaps, shadows, and detection flicker:
//
//   - Adaptive thresholding (THRESH_BINARY_INV + ADAPTIVE_GAUSSIAN_C) so the
//     classifier follows local lighting instead of a global black/white cutoff.
//   - Frame-to-frame latching: once a center line is acquired, subsequent
//     frames pick the candidate closest to the last tracked x. This stops the
//     robot from jumping onto a side line if it briefly looks more central.
//   - Look-ahead via a tall ROI: detection sees a vertical slice of road and
//     filters out short noise blobs (require height >= a fraction of the ROI).
//   - Decoupled command timer: /cmd_vel is published at a steady rate
//     regardless of camera FPS, so the wheels never starve.
//   - Brief coast on line loss: keep moving (slowed) for ~lost_timeout_s,
//     then stop. Catches small gaps without running off on a long miss.
//
// CPU OpenCV — the ROI is small enough that GPU round-trip dominates any
// kernel speedup, even on a Jetson.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

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

struct Candidate
{
  double cx;
  double cy;     // ROI-relative
  int height;
  int width;
  int area;
};

}  // namespace

class FollowLineNode : public rclcpp::Node
{
public:
  FollowLineNode()
  : Node("follow_line")
  {
    linear_speed_ = declare_parameter<double>("linear_speed", 0.15);
    max_angular_ = declare_parameter<double>("max_angular", 2.0);
    kp_ = declare_parameter<double>("kp", 1.8);
    kd_ = declare_parameter<double>("kd", 0.25);
    roi_top_frac_ = declare_parameter<double>("roi_top_frac", 0.55);
    roi_bottom_frac_ = declare_parameter<double>("roi_bottom_frac", 0.95);
    min_line_area_ = static_cast<int>(
      declare_parameter<int>("min_line_area", 200));
    max_line_width_frac_ = declare_parameter<double>("max_line_width_frac", 0.30);
    min_line_height_frac_ = declare_parameter<double>("min_line_height_frac", 0.30);
    max_track_jump_frac_ = declare_parameter<double>("max_track_jump_frac", 0.25);
    slow_on_turn_ = declare_parameter<double>("slow_on_turn", 0.5);
    lost_timeout_s_ = declare_parameter<double>("lost_timeout_s", 0.5);
    adaptive_block_ = static_cast<int>(
      declare_parameter<int>("adaptive_block", 41));
    if (adaptive_block_ < 3) {
      adaptive_block_ = 3;
    }
    if (adaptive_block_ % 2 == 0) {
      adaptive_block_ += 1;
    }
    adaptive_c_ = declare_parameter<double>("adaptive_c", 15.0);
    const double publish_rate = std::max(
      1.0, declare_parameter<double>("publish_rate", 20.0));
    debug_ = declare_parameter<bool>("debug", false);

    morph_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/line_follower/debug_image", 5);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/video_source/raw", rclcpp::QoS(1),
      std::bind(&FollowLineNode::onImage, this, std::placeholders::_1));

    cmd_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_rate)),
      std::bind(&FollowLineNode::publishCmd, this));

    last_detect_time_ = now();

    RCLCPP_INFO(
      get_logger(),
      "follow_line started (lin=%.2f ang_max=%.2f kp=%.2f kd=%.2f roi=[%.2f,%.2f] @ %.1f Hz).",
      linear_speed_, max_angular_, kp_, kd_,
      roi_top_frac_, roi_bottom_frac_, publish_rate);
  }

private:
  void onImage(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv::Mat bgr = imgmsg_to_bgr(*msg);
    if (bgr.empty()) {
      return;
    }
    const int W = bgr.cols;
    const int H = bgr.rows;
    const double image_center = W / 2.0;

    const int y0 = std::clamp(
      static_cast<int>(std::round(roi_top_frac_ * H)), 0, H - 2);
    const int y1 = std::clamp(
      static_cast<int>(std::round(roi_bottom_frac_ * H)), y0 + 1, H);
    cv::Mat roi = bgr.rowRange(y0, y1);
    const int roi_h = y1 - y0;

    cv::Mat gray, blurred, bin;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(
      blurred, bin, 255,
      cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
      adaptive_block_, adaptive_c_);
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, morph_kernel_);

    cv::Mat labels, stats, centroids;
    const int n_labels = cv::connectedComponentsWithStats(
      bin, labels, stats, centroids, 8);

    std::vector<Candidate> cands;
    cands.reserve(8);
    const int max_width = static_cast<int>(std::round(max_line_width_frac_ * W));
    const int min_height = static_cast<int>(std::round(min_line_height_frac_ * roi_h));

    for (int i = 1; i < n_labels; ++i) {
      const int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < min_line_area_) {
        continue;
      }
      const int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
      const int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
      if (width > max_width || height < min_height) {
        continue;
      }
      cands.push_back({
        centroids.at<double>(i, 0),
        centroids.at<double>(i, 1),
        height, width, area});
    }

    // Sort left-to-right.
    std::sort(cands.begin(), cands.end(),
      [](const Candidate & a, const Candidate & b) { return a.cx < b.cx; });

    // ---------- pick the center line ----------------------------------
    std::optional<double> target_x;
    if (!cands.empty()) {
      if (!have_track_) {
        if (cands.size() >= 3) {
          // We see three lanes — middle by x is the center one.
          target_x = cands[cands.size() / 2].cx;
        } else {
          // 1 or 2 candidates: pick the most central.
          double best_d = std::numeric_limits<double>::infinity();
          double best_x = 0.0;
          for (const auto & c : cands) {
            const double d = std::abs(c.cx - image_center);
            if (d < best_d) {
              best_d = d;
              best_x = c.cx;
            }
          }
          target_x = best_x;
        }
      } else {
        // Track: stay locked to whichever candidate is closest to where
        // we saw the center line last frame. Rejects sudden jumps that
        // would indicate the tracker latched onto a side line.
        double best_d = std::numeric_limits<double>::infinity();
        double best_x = 0.0;
        for (const auto & c : cands) {
          const double d = std::abs(c.cx - last_x_);
          if (d < best_d) {
            best_d = d;
            best_x = c.cx;
          }
        }
        const double max_jump = max_track_jump_frac_ * W;
        if (best_d <= max_jump) {
          target_x = best_x;
        }
      }
    }

    // ---------- control -----------------------------------------------
    const rclcpp::Time t_now = this->now();

    if (target_x.has_value()) {
      const double err = (*target_x - image_center) / image_center;  // [-1, 1]

      double derr = 0.0;
      if (have_track_) {
        const double dt = (t_now - last_detect_time_).seconds();
        if (dt > 0.0 && dt < 0.5) {
          derr = (err - last_err_) / dt;
        }
      }

      double w_cmd = -(kp_ * err + kd_ * derr);
      w_cmd = std::clamp(w_cmd, -max_angular_, max_angular_);

      double v_cmd = linear_speed_ * (1.0 - slow_on_turn_ * std::abs(err));
      v_cmd = std::max(0.0, v_cmd);

      latest_linear_ = v_cmd;
      latest_angular_ = w_cmd;

      last_x_ = *target_x;
      last_err_ = err;
      last_detect_time_ = t_now;
      have_track_ = true;
      coasting_ = false;
    } else if (have_track_) {
      const double since = (t_now - last_detect_time_).seconds();
      if (since < lost_timeout_s_) {
        // Brief coast: keep heading, halve speed so a real loss doesn't
        // carry us far.
        latest_linear_ = std::max(0.0, latest_linear_ * 0.5);
        coasting_ = true;
      } else {
        latest_linear_ = 0.0;
        latest_angular_ = 0.0;
        have_track_ = false;
        coasting_ = false;
      }
    } else {
      // Never acquired a line yet.
      latest_linear_ = 0.0;
      latest_angular_ = 0.0;
    }

    // ---------- debug visualization (lazy) -----------------------------
    const bool want_debug =
      debug_ || debug_pub_->get_subscription_count() > 0;
    if (want_debug) {
      publishDebug(bgr, y0, y1, cands, target_x, msg->header.stamp);
    }
  }

  void publishDebug(
    const cv::Mat & bgr, int y0, int y1,
    const std::vector<Candidate> & cands,
    std::optional<double> target_x,
    const builtin_interfaces::msg::Time & stamp)
  {
    const int W = bgr.cols;
    const int H = bgr.rows;
    const double image_center = W / 2.0;

    cv::Mat viz = bgr.clone();

    // Image-center reference (gray vertical).
    cv::line(viz, cv::Point(static_cast<int>(image_center), 0),
      cv::Point(static_cast<int>(image_center), H - 1),
      cv::Scalar(120, 120, 120), 1);

    // ROI rectangle.
    cv::rectangle(viz, cv::Point(0, y0), cv::Point(W - 1, y1 - 1),
      cv::Scalar(50, 200, 50), 1);

    // All candidate centroids (yellow), translated into full-image coords.
    for (const auto & c : cands) {
      const cv::Point p(
        static_cast<int>(std::round(c.cx)),
        static_cast<int>(std::round(c.cy)) + y0);
      cv::circle(viz, p, 4, cv::Scalar(0, 200, 255), -1);
    }

    // Selected target (red).
    if (target_x.has_value()) {
      const int tx = static_cast<int>(std::round(*target_x));
      cv::line(viz, cv::Point(tx, y0), cv::Point(tx, y1 - 1),
        cv::Scalar(0, 0, 255), 2);
      cv::circle(viz, cv::Point(tx, (y0 + y1) / 2), 8,
        cv::Scalar(0, 0, 255), 2);
    }

    char buf[160];
    const char * status = coasting_ ? "COAST"
      : (have_track_ ? "TRACK" : "SEARCH");
    std::snprintf(
      buf, sizeof(buf),
      "%s  lin=%.2f ang=%+.2f err=%+.2f n=%zu",
      status, latest_linear_, latest_angular_, last_err_, cands.size());
    cv::putText(viz, buf, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(viz, buf, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX,
      0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    debug_pub_->publish(bgr_to_imgmsg(viz, stamp, "line_follower"));
  }

  void publishCmd()
  {
    geometry_msgs::msg::Twist t;
    t.linear.x = latest_linear_;
    t.angular.z = latest_angular_;
    cmd_pub_->publish(t);
  }

  // ---- parameters ----
  double linear_speed_;
  double max_angular_;
  double kp_;
  double kd_;
  double roi_top_frac_;
  double roi_bottom_frac_;
  int min_line_area_;
  double max_line_width_frac_;
  double min_line_height_frac_;
  double max_track_jump_frac_;
  double slow_on_turn_;
  double lost_timeout_s_;
  int adaptive_block_;
  double adaptive_c_;
  bool debug_;

  // ---- state ----
  cv::Mat morph_kernel_;
  double last_x_ = 0.0;
  double last_err_ = 0.0;
  rclcpp::Time last_detect_time_;
  bool have_track_ = false;
  bool coasting_ = false;
  double latest_linear_ = 0.0;
  double latest_angular_ = 0.0;

  // ---- ROS ----
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FollowLineNode>());
  rclcpp::shutdown();
  return 0;
}
