// Single-line follower for the Puzzlebot.
//
// Detects ONE black line in a configurable ROI of the camera image and steers
// the robot to keep that line centered. Simpler than follow_line.cpp:
//   - no top-N picking
//   - no 3-line "middle by x" picker
//   - just: threshold-on-black → morphology → largest valid blob → centroid x
//
// Lateral control: P/PD on the normalized centroid error
//   err = (centroid_x - roi_center_x) / (roi_width / 2)   in [-1, +1]
//   omega = -(kp * err + kd * d_err/dt)
//
// Pipeline mirrors tools/single_line_tuner.py exactly so offline tuning
// transfers 1:1. When the tuner diverges from this file, fix both.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"

namespace
{

sensor_msgs::msg::CompressedImage bgr_to_compressed_jpeg(
  const cv::Mat & bgr, const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id, int quality)
{
  sensor_msgs::msg::CompressedImage msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;
  msg.format = "jpeg";
  std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
  cv::imencode(".jpg", bgr, msg.data, params);
  return msg;
}

int odd_at_least(int n, int lo)
{
  n = std::max(lo, n);
  return (n % 2 == 1) ? n : n + 1;
}

struct Blob
{
  double cx;
  double cy;      // ROI-relative
  int width;
  int height;
  int area;
  int bbox_x;     // full-image
  int bbox_y;     // full-image
};

}  // namespace

class FollowSingleLineNode : public rclcpp::Node
{
public:
  FollowSingleLineNode()
  : Node("follow_single_line")
  {
    linear_speed_ = declare_parameter<double>("linear_speed", 0.10);
    max_angular_ = declare_parameter<double>("max_angular", 2.5);
    kp_ = declare_parameter<double>("kp", 2.2);
    kd_ = declare_parameter<double>("kd", 0.25);
    slow_on_turn_ = std::clamp(
      declare_parameter<double>("slow_on_turn", 0.5), 0.0, 1.0);
    lost_timeout_s_ = declare_parameter<double>("lost_timeout_s", 0.5);
    // The camera is mounted slightly off-center: when the robot sits exactly
    // on the line, the centroid is NOT at the ROI center. center_offset is
    // that resting error (normalized to roi half-width, the same units the
    // HUD "err" shows). Calibrate: center the robot on a straight line by
    // hand, read err from the debug HUD, set center_offset to that value.
    center_offset_ = declare_parameter<double>("center_offset", 0.0);

    // YOLO-guided decisions once the line has been lost for lost_decision_s:
    //   left/right arrow -> stop following, odometry maneuver (drive
    //     inter_forward_m straight, turn 90 deg toward the arrow), resume.
    //   straight arrow   -> keep driving straight until the line returns.
    //   red/yellow light -> hold still until a green light is detected.
    //   nothing fresh    -> stop and wait for the line to reappear.
    lost_decision_s_ = declare_parameter<double>("lost_decision_s", 1.5);
    det_fresh_s_ = declare_parameter<double>("det_fresh_s", 4.0);
    det_min_conf_ = declare_parameter<double>("det_min_conf", 0.45);
    // Reject small bboxes (= far away signs/lights from OTHER intersections).
    // Area is the bbox fraction of the image: 0.01 ~ a 64x96 px box at 640x480.
    // Calibrate with the HUD: it shows the area of every accepted detection.
    det_min_area_ = declare_parameter<double>("det_min_area", 0.01);
    // Two signals of similar size (area ratio >= this) -> the rightmost wins.
    det_similar_ratio_ = declare_parameter<double>("det_similar_ratio", 0.7);
    // STOP sign: checked constantly (any mode). A fresh, big-enough "stop"
    // detection forces the robot to a halt; it resumes stop_hold_s after the
    // sign is last seen.
    stop_hold_s_ = declare_parameter<double>("stop_hold_s", 2.0);
    inter_forward_m_ = declare_parameter<double>("inter_forward_m", 0.20);
    inter_turn_rad_ =
      declare_parameter<double>("inter_turn_deg", 90.0) * M_PI / 180.0;
    turn_speed_ = declare_parameter<double>("turn_speed", 1.0);
    // After the turn, roll this far straight (odom) before handing control
    // back to the line follower, so the robot clears the intersection.
    exit_forward_m_ = declare_parameter<double>("exit_forward_m", 0.10);
    // Straight arrow: cross the intersection blind for this distance (odom)
    // before searching for the line again.
    straight_forward_m_ = declare_parameter<double>("straight_forward_m", 0.30);

    // Defaults below baked from tools/tune_line_detector outputs against a
    // recorded bag of the single-black-line track. Edit at runtime via
    // --ros-args -p ... if lighting / course changes.
    detect_scale_ = std::clamp(
      declare_parameter<double>("detect_scale", 0.48), 0.1, 1.0);

    roi_top_frac_ = declare_parameter<double>("roi_top_frac", 0.67);
    roi_bottom_frac_ = declare_parameter<double>("roi_bottom_frac", 1.00);
    roi_left_frac_ = std::clamp(
      declare_parameter<double>("roi_left_frac", 0.25), 0.0, 0.99);
    roi_right_frac_ = std::clamp(
      declare_parameter<double>("roi_right_frac", 0.73), 0.01, 1.0);

    gaussian_ksize_ = static_cast<int>(
      declare_parameter<int>("gaussian_ksize", 3));
    median_ksize_ = static_cast<int>(
      declare_parameter<int>("median_ksize", 0));
    black_threshold_ = std::clamp(
      static_cast<int>(declare_parameter<int>("black_threshold", 116)), 0, 255);
    open_kernel_ = std::max(
      1, static_cast<int>(declare_parameter<int>("open_kernel", 8)));
    close_kernel_h_ = std::max(
      1, static_cast<int>(declare_parameter<int>("close_kernel_h", 11)));

    min_line_area_ = static_cast<int>(
      declare_parameter<int>("min_line_area", 733));
    max_line_width_frac_ = declare_parameter<double>("max_line_width_frac", 0.72);
    min_line_height_frac_ = declare_parameter<double>("min_line_height_frac", 0.62);

    const double publish_rate = std::max(
      1.0, declare_parameter<double>("publish_rate", 20.0));

    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_jpeg_quality_ = std::clamp(
      static_cast<int>(declare_parameter<int>("debug_jpeg_quality", 70)),
      1, 100);

    const std::string input_topic = declare_parameter<std::string>(
      "input_topic", "/camera/image_rect/compressed");

    // Traffic-light gating. Subscribes to /traffic_light/go (Bool) from the
    // traffic_light node. Yellow/red are reported as go=false, green/none
    // as go=true, so a simple gate is enough.
    //  - enable_traffic_light=false → ignore the topic entirely
    //  - no message received yet     → default to "go" so the robot
    //    isn't bricked when the detector is off / late to start
    enable_traffic_light_ = declare_parameter<bool>("enable_traffic_light", true);
    const std::string go_topic = declare_parameter<std::string>(
      "traffic_light_topic", "/traffic_light/go");

    cv::setNumThreads(1);
    rebuildKernels();

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    // Normalized lateral error (after center_offset), one sample per camera
    // frame with the line visible. Plot it:  rqt_plot /line_follower/error/data
    err_pub_ = create_publisher<std_msgs::msg::Float32>(
      "/line_follower/error", 10);
    if (publish_debug_image_) {
      debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/line_follower/debug_image/compressed", 5);
    }

    image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      input_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&FollowSingleLineNode::onImage, this, std::placeholders::_1));

    if (enable_traffic_light_) {
      go_sub_ = create_subscription<std_msgs::msg::Bool>(
        go_topic, 10,
        std::bind(&FollowSingleLineNode::onGo, this, std::placeholders::_1));
    }

    det_sub_ = create_subscription<std_msgs::msg::String>(
      declare_parameter<std::string>("detections_topic", "/yolo/detections"),
      10,
      std::bind(&FollowSingleLineNode::onDetections, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      declare_parameter<std::string>("odom_topic", "/odom"), 20,
      std::bind(&FollowSingleLineNode::onOdom, this, std::placeholders::_1));

    cmd_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_rate)),
      std::bind(&FollowSingleLineNode::publishCmd, this));

    last_detect_time_ = now();
    last_line_seen_time_ = now();
    arrow_time_ = now();
    light_time_ = now();
    stop_seen_time_ = now();

    // Live tuning of the control/decision params (ros2 param set ...) so the
    // oscillation/offset calibration doesn't need a node restart.
    param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {
        rcl_interfaces::msg::SetParametersResult res;
        res.successful = true;
        for (const auto & p : ps) {
          const auto & n = p.get_name();
          if (n == "kp") {kp_ = p.as_double();}
          else if (n == "kd") {kd_ = p.as_double();}
          else if (n == "center_offset") {center_offset_ = p.as_double();}
          else if (n == "linear_speed") {linear_speed_ = p.as_double();}
          else if (n == "max_angular") {max_angular_ = p.as_double();}
          else if (n == "slow_on_turn") {
            slow_on_turn_ = std::clamp(p.as_double(), 0.0, 1.0);
          } else if (n == "turn_speed") {turn_speed_ = p.as_double();}
          else if (n == "lost_decision_s") {lost_decision_s_ = p.as_double();}
          else if (n == "det_min_conf") {det_min_conf_ = p.as_double();}
          else if (n == "det_min_area") {det_min_area_ = p.as_double();}
          else if (n == "det_similar_ratio") {det_similar_ratio_ = p.as_double();}
          else if (n == "stop_hold_s") {stop_hold_s_ = p.as_double();}
          else if (n == "det_fresh_s") {det_fresh_s_ = p.as_double();}
          else if (n == "inter_forward_m") {inter_forward_m_ = p.as_double();}
          else if (n == "exit_forward_m") {exit_forward_m_ = p.as_double();}
          else if (n == "straight_forward_m") {straight_forward_m_ = p.as_double();}
          else if (n == "black_threshold") {
            black_threshold_ = std::clamp(static_cast<int>(p.as_int()), 0, 255);
          }
        }
        return res;
      });

    RCLCPP_INFO(
      get_logger(),
      "follow_single_line started "
      "(lin=%.2f ang_max=%.2f kp=%.2f kd=%.2f roi=[%.2f,%.2f]x[%.2f,%.2f] "
      "black=%d @ %.0f Hz).",
      linear_speed_, max_angular_, kp_, kd_,
      roi_left_frac_, roi_right_frac_, roi_top_frac_, roi_bottom_frac_,
      black_threshold_, publish_rate);
  }

private:
  void rebuildKernels()
  {
    const int ok = std::max(1, open_kernel_);
    open_k_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ok, ok));
    close_k_ = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(3, std::max(1, close_kernel_h_)));
  }

  void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
  {
    if (msg->data.empty()) {
      return;
    }
    cv::Mat bgr = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      return;
    }
    if (detect_scale_ < 0.999) {
      cv::Mat small;
      cv::resize(
        bgr, small, cv::Size(), detect_scale_, detect_scale_, cv::INTER_AREA);
      bgr = small;
    }
    const int W = bgr.cols;
    const int H = bgr.rows;

    const int x0 = std::clamp(
      static_cast<int>(std::round(roi_left_frac_ * W)), 0, W - 2);
    const int x1 = std::clamp(
      static_cast<int>(std::round(roi_right_frac_ * W)), x0 + 1, W);
    const int y0 = std::clamp(
      static_cast<int>(std::round(roi_top_frac_ * H)), 0, H - 2);
    const int y1 = std::clamp(
      static_cast<int>(std::round(roi_bottom_frac_ * H)), y0 + 1, H);
    cv::Mat roi = bgr(cv::Rect(x0, y0, x1 - x0, y1 - y0));
    const int roi_w = x1 - x0;
    const int roi_h = y1 - y0;
    const double roi_center_x = (x0 + x1) / 2.0;

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    const int gk = odd_at_least(gaussian_ksize_, 1);
    if (gk >= 3) {
      cv::GaussianBlur(gray, gray, cv::Size(gk, gk), 0);
    }
    const int mk = odd_at_least(median_ksize_, 1);
    if (mk >= 3) {
      cv::medianBlur(gray, gray, mk);
    }

    cv::Mat mask;
    cv::threshold(
      gray, mask, static_cast<double>(black_threshold_), 255.0,
      cv::THRESH_BINARY_INV);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_k_);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_k_);

    cv::Mat labels, stats, centroids;
    const int n_labels = cv::connectedComponentsWithStats(
      mask, labels, stats, centroids, 8);

    const int max_width = static_cast<int>(std::round(max_line_width_frac_ * roi_w));
    const int min_height = static_cast<int>(std::round(min_line_height_frac_ * roi_h));

    std::optional<Blob> best;
    for (int i = 1; i < n_labels; ++i) {
      const int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < min_line_area_) {
        continue;
      }
      const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
      const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
      if (w > max_width || h < min_height) {
        continue;
      }
      Blob b{
        centroids.at<double>(i, 0) + x0,
        centroids.at<double>(i, 1),
        w, h, area,
        stats.at<int>(i, cv::CC_STAT_LEFT) + x0,
        stats.at<int>(i, cv::CC_STAT_TOP) + y0,
      };
      if (!best.has_value() || b.area > best->area) {
        best = b;
      }
    }

    const rclcpp::Time t_now = this->now();
    std::optional<double> target_x;
    if (best.has_value()) {
      target_x = best->cx;
    }

    if (mode_ == Mode::FOLLOW && target_x.has_value()) {
      // Normalize error to roi half-width so kp/kd stay scale-independent.
      // center_offset cancels the camera's lateral mounting bias.
      const double half = std::max(1.0, roi_w / 2.0);
      const double err = (*target_x - roi_center_x) / half - center_offset_;

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
      last_err_ = err;
      last_detect_time_ = t_now;
      last_line_seen_time_ = t_now;
      have_track_ = true;
      coasting_ = false;
      search_stop_ = false;
      straight_latch_ = false;

      std_msgs::msg::Float32 err_msg;
      err_msg.data = static_cast<float>(err);
      err_pub_->publish(err_msg);
    } else if (mode_ == Mode::FOLLOW) {
      // Line lost (or never acquired): STOP and stand still. After
      // lost_decision_s, publishCmd() decides from the YOLO detections.
      // Exception: a prior "straight arrow" decision latches driving
      // straight until the line is found again.
      latest_linear_ = straight_latch_ ? linear_speed_ : 0.0;
      latest_angular_ = 0.0;
      coasting_ = have_track_;  // purely a debug-overlay status hint
    }

    if (debug_pub_ && debug_pub_->get_subscription_count() > 0) {
      publishDebug(bgr, x0, y0, x1, y1, best, target_x, msg->header.stamp);
    }
  }

  void publishDebug(
    const cv::Mat & bgr, int x0, int y0, int x1, int y1,
    const std::optional<Blob> & best, std::optional<double> target_x,
    const builtin_interfaces::msg::Time & stamp)
  {
    cv::Mat viz = bgr.clone();
    const int W = viz.cols;
    const int H = viz.rows;
    (void)W; (void)H;
    const double roi_center_x = (x0 + x1) / 2.0;

    cv::rectangle(viz, cv::Point(x0, y0), cv::Point(x1 - 1, y1 - 1),
      cv::Scalar(50, 200, 50), 1);
    cv::line(viz, cv::Point(static_cast<int>(roi_center_x), y0),
      cv::Point(static_cast<int>(roi_center_x), y1 - 1),
      cv::Scalar(120, 120, 120), 1);
    // The actual setpoint (= geometric center + camera mounting offset).
    // Steering drives the line centroid onto THIS magenta line, not the gray.
    const double setpoint_x =
      roi_center_x + center_offset_ * ((x1 - x0) / 2.0);
    cv::line(viz, cv::Point(static_cast<int>(setpoint_x), y0),
      cv::Point(static_cast<int>(setpoint_x), y1 - 1),
      cv::Scalar(255, 0, 255), 1);

    if (best.has_value()) {
      cv::rectangle(
        viz, cv::Point(best->bbox_x, best->bbox_y),
        cv::Point(best->bbox_x + best->width, best->bbox_y + best->height),
        cv::Scalar(0, 200, 255), 2);
      cv::circle(
        viz,
        cv::Point(
          static_cast<int>(std::round(best->cx)),
          static_cast<int>(std::round(best->cy)) + y0),
        5, cv::Scalar(0, 200, 255), -1);
    }
    if (target_x.has_value()) {
      const int tx = static_cast<int>(std::round(*target_x));
      cv::line(viz, cv::Point(tx, y0), cv::Point(tx, y1 - 1),
        cv::Scalar(0, 0, 255), 2);
      cv::circle(viz, cv::Point(tx, (y0 + y1) / 2), 8,
        cv::Scalar(0, 0, 255), 2);
    }

    const rclcpp::Time t_now = this->now();
    const auto put_line = [&viz](const char * txt, int y, cv::Scalar color) {
        cv::putText(viz, txt, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
          0.55, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(viz, txt, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX,
          0.55, color, 1, cv::LINE_AA);
      };

    // Line 1: mode + control numbers.
    char buf[160];
    const char * status =
      mode_ == Mode::HOLD_RED ? "HOLD-RED" :
      mode_ == Mode::INTER_FORWARD ? "INT-FWD" :
      mode_ == Mode::INTER_TURN ? "INT-TURN" :
      mode_ == Mode::INTER_EXIT ? "INT-EXIT" :
      mode_ == Mode::STRAIGHT_FWD ? "STR-FWD" :
      search_stop_ ? "WAIT-LINE" :
      straight_latch_ ? "STRAIGHT" :
      coasting_ ? "COAST" : (have_track_ ? "TRACK" : "SEARCH");
    std::snprintf(
      buf, sizeof(buf),
      "%s  lin=%.2f ang=%+.2f err=%+.2f off=%+.2f area=%d",
      status, latest_linear_, latest_angular_, last_err_, center_offset_,
      best.has_value() ? best->area : 0);
    const cv::Scalar mode_color =
      mode_ == Mode::HOLD_RED ? cv::Scalar(0, 0, 255) :
      (mode_ == Mode::INTER_FORWARD || mode_ == Mode::INTER_TURN) ?
        cv::Scalar(0, 255, 255) :
      search_stop_ ? cv::Scalar(0, 165, 255) : cv::Scalar(255, 255, 255);
    put_line(buf, 22, mode_color);

    // Line 2: what YOLO last saw (and how stale it is) + lost-line timer.
    const char * arrow_name =
      arrow_ == Arrow::LEFT ? "left" :
      arrow_ == Arrow::RIGHT ? "right" :
      arrow_ == Arrow::STRAIGHT ? "straight" : "-";
    const char * light_name =
      light_ == Light::RED ? "RED" :
      light_ == Light::YELLOW ? "YELLOW" :
      light_ == Light::GREEN ? "GREEN" : "-";
    const double lost_s =
      have_track_ ? (t_now - last_line_seen_time_).seconds() : 0.0;
    std::snprintf(
      buf, sizeof(buf),
      "yolo: arrow=%s(%.2f a%.3f %.1fs)  light=%s(%.2f a%.3f %.1fs)  lost=%.1f/%.1fs",
      arrow_name, arrow_conf_, arrow_area_,
      arrow_ == Arrow::NONE ? 0.0 : (t_now - arrow_time_).seconds(),
      light_name, light_conf_, light_area_,
      light_ == Light::NONE ? 0.0 : (t_now - light_time_).seconds(),
      lost_s, lost_decision_s_);
    put_line(buf, 44, cv::Scalar(255, 255, 255));

    if (stopSignActive(t_now)) {
      std::snprintf(
        buf, sizeof(buf), "STOP SIGN (a%.3f) - halted %.1f/%.1fs",
        stop_area_, (t_now - stop_seen_time_).seconds(), stop_hold_s_);
      put_line(buf, 88, cv::Scalar(0, 0, 255));
    }

    // Line 3: live maneuver progress / hold reason.
    if (mode_ == Mode::INTER_FORWARD) {
      std::snprintf(
        buf, sizeof(buf), "maneuver: fwd %.2f / %.2f m  then 90deg %s",
        have_odom_ ? std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_) : 0.0,
        inter_forward_m_, turn_dir_ > 0 ? "LEFT" : "RIGHT");
      put_line(buf, 66, cv::Scalar(0, 255, 255));
    } else if (mode_ == Mode::INTER_TURN) {
      std::snprintf(
        buf, sizeof(buf), "maneuver: turn %.0f / %.0f deg %s",
        std::abs(turn_accum_) * 180.0 / M_PI, inter_turn_rad_ * 180.0 / M_PI,
        turn_dir_ > 0 ? "LEFT" : "RIGHT");
      put_line(buf, 66, cv::Scalar(0, 255, 255));
    } else if (mode_ == Mode::INTER_EXIT) {
      std::snprintf(
        buf, sizeof(buf), "maneuver: exit fwd %.2f / %.2f m",
        have_odom_ ? std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_) : 0.0,
        exit_forward_m_);
      put_line(buf, 66, cv::Scalar(0, 255, 255));
    } else if (mode_ == Mode::STRAIGHT_FWD) {
      std::snprintf(
        buf, sizeof(buf), "maneuver: straight fwd %.2f / %.2f m",
        have_odom_ ? std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_) : 0.0,
        straight_forward_m_);
      put_line(buf, 66, cv::Scalar(0, 255, 255));
    } else if (mode_ == Mode::HOLD_RED) {
      put_line("waiting for GREEN light...", 66, cv::Scalar(0, 0, 255));
    } else if (search_stop_) {
      put_line("stopped: no line, no fresh detection", 66,
        cv::Scalar(0, 165, 255));
    }

    debug_pub_->publish(
      bgr_to_compressed_jpeg(
        viz, stamp, "follow_single_line", debug_jpeg_quality_));
  }

  void onGo(std_msgs::msg::Bool::ConstSharedPtr msg)
  {
    const bool prev = traffic_light_go_;
    traffic_light_go_ = msg->data;
    have_go_ = true;
    if (prev != traffic_light_go_) {
      RCLCPP_INFO(
        get_logger(), "traffic_light go=%s",
        traffic_light_go_ ? "true" : "false");
    }
  }

  // "/yolo/detections": comma list of name:conf:area:cx entries (area = bbox
  // fraction of the image, cx = normalized bbox center x, e.g.
  // "left:0.87:0.0231:0.62"). Small bboxes are far away (another
  // intersection's light/sign) and get rejected. When two signals of the
  // same group (arrows / lights) have SIMILAR size (area ratio >=
  // det_similar_ratio) the one more to the RIGHT wins — that's the signal
  // for our lane. Keeps the freshest arrow / traffic light / stop sign.
  void onDetections(std_msgs::msg::String::ConstSharedPtr msg)
  {
    const rclcpp::Time t_now = this->now();
    struct Cand { int kind; double conf, area, cx; };
    std::vector<Cand> arrows;
    std::vector<Cand> lights;
    std::stringstream ss(msg->data);
    std::string item;
    while (std::getline(ss, item, ',')) {
      // split name:conf[:area[:cx]] (missing -> permissive defaults)
      const auto c1 = item.find(':');
      if (c1 == std::string::npos) {
        continue;
      }
      const auto c2 = item.find(':', c1 + 1);
      const auto c3 = c2 == std::string::npos
        ? std::string::npos : item.find(':', c2 + 1);
      const std::string name = item.substr(0, c1);
      const double conf = std::atof(item.c_str() + c1 + 1);
      const double area =
        c2 == std::string::npos ? 1.0 : std::atof(item.c_str() + c2 + 1);
      const double cx =
        c3 == std::string::npos ? 0.5 : std::atof(item.c_str() + c3 + 1);
      if (conf < det_min_conf_ || area < det_min_area_) {
        continue;
      }
      if (name == "stop") {
        stop_seen_time_ = t_now;
        stop_area_ = area;
        have_stop_seen_ = true;
      } else if (name == "left") {
        arrows.push_back({static_cast<int>(Arrow::LEFT), conf, area, cx});
      } else if (name == "right") {
        arrows.push_back({static_cast<int>(Arrow::RIGHT), conf, area, cx});
      } else if (name == "straight") {
        arrows.push_back({static_cast<int>(Arrow::STRAIGHT), conf, area, cx});
      } else if (name == "traffic_light_red") {
        lights.push_back({static_cast<int>(Light::RED), conf, area, cx});
      } else if (name == "traffic_light_yellow") {
        lights.push_back({static_cast<int>(Light::YELLOW), conf, area, cx});
      } else if (name == "traffic_light_green") {
        lights.push_back({static_cast<int>(Light::GREEN), conf, area, cx});
      }
    }

    // Pick the biggest; if others are similar-sized, the rightmost wins.
    const auto pick = [this](const std::vector<Cand> & v) -> int {
        if (v.empty()) {
          return -1;
        }
        double a_max = 0.0;
        for (const auto & c : v) {
          a_max = std::max(a_max, c.area);
        }
        int best = -1;
        for (size_t i = 0; i < v.size(); ++i) {
          if (v[i].area >= det_similar_ratio_ * a_max &&
            (best < 0 || v[i].cx > v[static_cast<size_t>(best)].cx))
          {
            best = static_cast<int>(i);
          }
        }
        return best;
      };

    const int ai = pick(arrows);
    if (ai >= 0) {
      arrow_ = static_cast<Arrow>(arrows[ai].kind);
      arrow_conf_ = arrows[ai].conf;
      arrow_area_ = arrows[ai].area;
      arrow_time_ = t_now;
      if (arrows.size() > 1) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%zu arrows seen -> took rightmost (cx=%.2f)",
          arrows.size(), arrows[ai].cx);
      }
    }
    const int li = pick(lights);
    if (li >= 0) {
      light_ = static_cast<Light>(lights[li].kind);
      light_conf_ = lights[li].conf;
      light_area_ = lights[li].area;
      light_time_ = t_now;
      if (lights.size() > 1) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "%zu lights seen -> took rightmost (cx=%.2f)",
          lights.size(), lights[li].cx);
      }
    }
  }

  bool stopSignActive(const rclcpp::Time & t_now) const
  {
    return have_stop_seen_ &&
           (t_now - stop_seen_time_).seconds() < stop_hold_s_;
  }

  void onOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    odom_x_ = msg->pose.pose.position.x;
    odom_y_ = msg->pose.pose.position.y;
    const auto & q = msg->pose.pose.orientation;
    const double yaw = std::atan2(
      2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    if (mode_ == Mode::INTER_TURN && have_odom_) {
      // Accumulate wrapped yaw increments so the 90 deg target is robust.
      turn_accum_ += std::atan2(std::sin(yaw - prev_yaw_), std::cos(yaw - prev_yaw_));
    }
    prev_yaw_ = yaw;
    have_odom_ = true;
  }

  // Line has been lost for > lost_decision_s while following: pick what to
  // do from the freshest YOLO detections.
  void decideOnLost(const rclcpp::Time & t_now)
  {
    const bool light_fresh =
      light_ != Light::NONE && (t_now - light_time_).seconds() < det_fresh_s_;
    const bool arrow_fresh =
      arrow_ != Arrow::NONE && (t_now - arrow_time_).seconds() < det_fresh_s_;

    if (light_fresh && light_ != Light::GREEN) {
      RCLCPP_INFO(
        get_logger(), "line lost + %s light -> holding until green",
        light_ == Light::RED ? "red" : "yellow");
      mode_ = Mode::HOLD_RED;
    } else if (arrow_fresh && (arrow_ == Arrow::LEFT || arrow_ == Arrow::RIGHT)) {
      if (!have_odom_) {
        RCLCPP_WARN(
          get_logger(),
          "arrow seen but no /odom yet -> stopping (cannot run the maneuver)");
        search_stop_ = true;
        return;
      }
      turn_dir_ = (arrow_ == Arrow::LEFT) ? 1.0 : -1.0;
      fwd_x0_ = odom_x_;
      fwd_y0_ = odom_y_;
      mode_ = Mode::INTER_FORWARD;
      arrow_ = Arrow::NONE;  // consume: don't reuse it for a second maneuver
      RCLCPP_INFO(
        get_logger(),
        "line lost + %s arrow -> forward %.2f m, then 90 deg %s",
        turn_dir_ > 0 ? "left" : "right", inter_forward_m_,
        turn_dir_ > 0 ? "left" : "right");
    } else if (arrow_fresh && arrow_ == Arrow::STRAIGHT) {
      arrow_ = Arrow::NONE;  // consume
      if (have_odom_) {
        // Cross the intersection blind (odom-measured), then keep going
        // straight until the line is reacquired.
        fwd_x0_ = odom_x_;
        fwd_y0_ = odom_y_;
        mode_ = Mode::STRAIGHT_FWD;
        RCLCPP_INFO(
          get_logger(),
          "line lost + straight arrow -> forward %.2f m, then search the line",
          straight_forward_m_);
      } else {
        straight_latch_ = true;
        RCLCPP_INFO(
          get_logger(),
          "line lost + straight arrow (no /odom) -> driving straight until the line is back");
      }
    } else {
      search_stop_ = true;
      RCLCPP_INFO(
        get_logger(),
        "line lost %.1f s, no fresh YOLO detection -> stopping until the line is back",
        (t_now - last_line_seen_time_).seconds());
    }
  }

  void publishCmd()
  {
    const rclcpp::Time t_now = this->now();
    geometry_msgs::msg::Twist t;  // zero-initialized: default is full stop

    // STOP sign override: checked constantly, beats every mode. The robot
    // halts in place and resumes stop_hold_s after the sign was last seen
    // (the state machine stays where it was, so a paused maneuver continues).
    if (stopSignActive(t_now)) {
      cmd_pub_->publish(t);
      return;
    }

    switch (mode_) {
      case Mode::HOLD_RED:
        // Stopped at a red/yellow light: only a fresh green releases us.
        if (light_ == Light::GREEN &&
          (t_now - light_time_).seconds() < det_fresh_s_)
        {
          RCLCPP_INFO(get_logger(), "green light -> resuming");
          mode_ = Mode::FOLLOW;
          last_line_seen_time_ = t_now;
        }
        break;

      case Mode::INTER_FORWARD: {
        if (!have_odom_) {
          break;  // stay stopped; cannot measure distance
        }
        const double d = std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_);
        if (d >= inter_forward_m_) {
          turn_accum_ = 0.0;
          mode_ = Mode::INTER_TURN;
          RCLCPP_INFO(
            get_logger(), "intersection: %.2f m done, turning %s",
            d, turn_dir_ > 0 ? "left" : "right");
        } else {
          t.linear.x = linear_speed_;
        }
        break;
      }

      case Mode::INTER_TURN:
        if (std::abs(turn_accum_) >= inter_turn_rad_) {
          // Turn done -> short straight exit segment (odom-measured).
          fwd_x0_ = odom_x_;
          fwd_y0_ = odom_y_;
          mode_ = Mode::INTER_EXIT;
          RCLCPP_INFO(
            get_logger(),
            "intersection: turn done (%.0f deg) -> exiting straight %.2f m",
            std::abs(turn_accum_) * 180.0 / M_PI, exit_forward_m_);
        } else {
          t.angular.z = turn_dir_ * turn_speed_;
        }
        break;

      case Mode::STRAIGHT_FWD: {
        if (!have_odom_) {
          break;  // stay stopped; cannot measure distance
        }
        const double d = std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_);
        if (d >= straight_forward_m_) {
          mode_ = Mode::FOLLOW;
          have_track_ = false;
          last_err_ = 0.0;
          latest_linear_ = linear_speed_;
          latest_angular_ = 0.0;
          last_line_seen_time_ = t_now;
          straight_latch_ = true;  // keep rolling until the line is found
          RCLCPP_INFO(
            get_logger(),
            "straight: %.2f m done -> searching the line", d);
        } else {
          t.linear.x = linear_speed_;
        }
        break;
      }

      case Mode::INTER_EXIT: {
        if (!have_odom_) {
          break;  // stay stopped; cannot measure distance
        }
        const double d = std::hypot(odom_x_ - fwd_x0_, odom_y_ - fwd_y0_);
        if (d >= exit_forward_m_) {
          mode_ = Mode::FOLLOW;
          have_track_ = false;
          last_err_ = 0.0;
          latest_linear_ = linear_speed_;
          latest_angular_ = 0.0;
          last_line_seen_time_ = t_now;
          // Keep driving straight until the line is reacquired (otherwise
          // the stop-on-lost logic would freeze us right here).
          straight_latch_ = true;
          RCLCPP_INFO(
            get_logger(), "intersection: exit done (%.2f m) -> following again", d);
        } else {
          t.linear.x = linear_speed_;
        }
        break;
      }

      case Mode::FOLLOW: {
        const bool line_lost =
          (t_now - last_line_seen_time_).seconds() > lost_decision_s_;
        if (have_track_ && line_lost && !straight_latch_ && !search_stop_) {
          decideOnLost(t_now);
        }
        if (mode_ == Mode::FOLLOW && !search_stop_) {
          t.linear.x = latest_linear_;
          t.angular.z = latest_angular_;
        }
        break;
      }
    }

    // Traffic-light gate runs LAST so it overrides whatever the state
    // machine decided. go=false → full stop until we see go=true again.
    const bool gate_stop =
      enable_traffic_light_ && have_go_ && !traffic_light_go_;
    if (gate_stop) {
      t.linear.x = 0.0;
      t.angular.z = 0.0;
    }
    cmd_pub_->publish(t);
  }

  // ---- params ----
  double linear_speed_;
  double max_angular_;
  double kp_;
  double kd_;
  double slow_on_turn_;
  double lost_timeout_s_;
  double center_offset_;
  double detect_scale_;
  double roi_top_frac_;
  double roi_bottom_frac_;
  double roi_left_frac_;
  double roi_right_frac_;
  int gaussian_ksize_;
  int median_ksize_;
  int black_threshold_;
  int open_kernel_;
  int close_kernel_h_;
  int min_line_area_;
  double max_line_width_frac_;
  double min_line_height_frac_;
  bool publish_debug_image_;
  int debug_jpeg_quality_;
  bool enable_traffic_light_;

  // ---- lost-line decision params ----
  double lost_decision_s_;
  double det_fresh_s_;
  double det_min_conf_;
  double det_min_area_;
  double det_similar_ratio_;
  double stop_hold_s_;
  double inter_forward_m_;
  double inter_turn_rad_;
  double turn_speed_;
  double exit_forward_m_;
  double straight_forward_m_;

  // ---- state ----
  enum class Mode {
    FOLLOW, HOLD_RED, INTER_FORWARD, INTER_TURN, INTER_EXIT, STRAIGHT_FWD
  };
  enum class Arrow { NONE, LEFT, RIGHT, STRAIGHT };
  enum class Light { NONE, RED, YELLOW, GREEN };

  cv::Mat open_k_;
  cv::Mat close_k_;
  double last_err_ = 0.0;
  rclcpp::Time last_detect_time_;
  bool have_track_ = false;
  bool coasting_ = false;
  double latest_linear_ = 0.0;
  double latest_angular_ = 0.0;
  bool have_go_ = false;
  bool traffic_light_go_ = true;

  Mode mode_ = Mode::FOLLOW;
  Arrow arrow_ = Arrow::NONE;
  Light light_ = Light::NONE;
  double arrow_conf_ = 0.0;
  double light_conf_ = 0.0;
  double arrow_area_ = 0.0;
  double light_area_ = 0.0;
  double stop_area_ = 0.0;
  bool have_stop_seen_ = false;
  rclcpp::Time arrow_time_;
  rclcpp::Time light_time_;
  rclcpp::Time stop_seen_time_;
  rclcpp::Time last_line_seen_time_;
  bool search_stop_ = false;
  bool straight_latch_ = false;
  bool have_odom_ = false;
  double odom_x_ = 0.0;
  double odom_y_ = 0.0;
  double prev_yaw_ = 0.0;
  double turn_accum_ = 0.0;
  double fwd_x0_ = 0.0;
  double fwd_y0_ = 0.0;
  double turn_dir_ = 1.0;  // +1 left (CCW), -1 right (CW)

  // ---- ROS ----
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr go_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr det_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr err_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FollowSingleLineNode>());
  rclcpp::shutdown();
  return 0;
}
