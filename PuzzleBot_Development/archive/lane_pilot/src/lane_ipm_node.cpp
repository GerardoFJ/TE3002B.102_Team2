// lane_ipm_node — perception half of the metric lane follower.
//
//   /camera/image_rect/compressed  (JPEG, rectified 1280x720)
//        -> decode -> IPM warp to metric bird's-eye
//        -> detect the middle line of the 3-line lane
//        -> /lane/points          (geometry_msgs/PoseArray, frame=base_link)
//        -> /lane/ipm_debug/compressed   (lazy bird's-eye overlay)
//
// Each published pose is a centerline point in meters in base_link (z=0). The
// downstream lane_pilot_node accumulates these in the odom frame (filling the
// camera blind spot) and tracks them with pure pursuit.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "geometry_msgs/msg/pose_array.hpp"

#include "ipm.hpp"
#include "lane_detector.hpp"

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

}  // namespace

class LaneIpmNode : public rclcpp::Node
{
public:
  LaneIpmNode()
  : Node("lane_ipm_node"),
    detector_(lane_pilot::DetectorParams{})
  {
    homography_file_ = declare_parameter<std::string>("homography_file", "");
    const std::string input_topic = declare_parameter<std::string>(
      "input_topic", "/camera/image_rect/compressed");
    const std::string output_topic = declare_parameter<std::string>(
      "output_topic", "/lane/points");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

    lane_pilot::BirdViewSpec spec;
    spec.x_min = declare_parameter<double>("bird_x_min", 0.05);
    spec.x_max = declare_parameter<double>("bird_x_max", 0.70);
    spec.y_half = declare_parameter<double>("bird_y_half", 0.35);
    spec.mpp = declare_parameter<double>("bird_mpp", 0.0025);

    dp_.black_threshold = declare_parameter<int>("black_threshold", 110);
    dp_.invert = declare_parameter<bool>("invert_threshold", true);
    dp_.blur_ksize = declare_parameter<int>("blur_ksize", 3);
    dp_.open_px = declare_parameter<int>("open_px", 0);
    dp_.close_px = declare_parameter<int>("close_px", 0);
    dp_.min_cluster_px = declare_parameter<int>("min_cluster_px", 6);
    dp_.max_cluster_px = declare_parameter<int>("max_cluster_px", 60);
    dp_.row_step = std::max(1, static_cast<int>(declare_parameter<int>("row_step", 2)));
    dp_.max_jump_m = declare_parameter<double>("max_jump_m", 0.06);
    dp_.min_points = declare_parameter<int>("min_points", 6);
    dp_.roi_x_min = declare_parameter<double>("roi_x_min", 0.05);
    dp_.roi_x_max = declare_parameter<double>("roi_x_max", 0.55);
    dp_.roi_y_half = declare_parameter<double>("roi_y_half", 0.25);
    raw_top_frac_ = declare_parameter<double>("raw_top_frac", 0.0);
    raw_bottom_frac_ = declare_parameter<double>("raw_bottom_frac", 1.0);
    raw_left_frac_ = declare_parameter<double>("raw_left_frac", 0.0);
    raw_right_frac_ = declare_parameter<double>("raw_right_frac", 1.0);
    detector_.set_params(dp_);

    // Live tuning: ros2 param set /lane_ipm_node <name> <value> updates the
    // detector immediately (no rebuild) — watch /lane/ipm_debug while tuning.
    param_cb_handle_ = add_on_set_parameters_callback(
      std::bind(&LaneIpmNode::onSetParams, this, std::placeholders::_1));

    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_jpeg_quality_ = std::clamp(
      static_cast<int>(declare_parameter<int>("debug_jpeg_quality", 70)), 1, 100);

    cv::setNumThreads(1);

    ipm_.set_view(spec);
    if (!homography_file_.empty()) {
      std::string err;
      if (!ipm_.load(homography_file_, &err)) {
        RCLCPP_ERROR(
          get_logger(), "Failed to load homography: %s. Node will idle until "
          "a valid 'homography_file' is provided.", err.c_str());
      } else {
        RCLCPP_INFO(
          get_logger(), "Loaded ground homography from %s",
          homography_file_.c_str());
      }
    } else {
      RCLCPP_ERROR(
        get_logger(), "No 'homography_file' parameter set — run "
        "tools/calibrate_ground_homography.py first. Node will idle.");
    }

    points_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(output_topic, 10);
    if (publish_debug_image_) {
      debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/lane/ipm_debug/compressed", 5);
    }
    image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      input_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&LaneIpmNode::onImage, this, std::placeholders::_1));

    stats_timer_ = create_wall_timer(
      std::chrono::seconds(5), std::bind(&LaneIpmNode::logStats, this));

    RCLCPP_INFO(
      get_logger(),
      "lane_ipm_node started (bird %dx%d px, X=[%.2f,%.2f] Y=+/-%.2f, "
      "mpp=%.4f, thr=%d).",
      spec.width(), spec.height(), spec.x_min, spec.x_max, spec.y_half,
      spec.mpp, dp_.black_threshold);
  }

private:
  rcl_interfaces::msg::SetParametersResult onSetParams(
    const std::vector<rclcpp::Parameter> & params)
  {
    for (const auto & p : params) {
      const std::string & n = p.get_name();
      if (n == "black_threshold") {dp_.black_threshold = static_cast<int>(p.as_int());}
      else if (n == "invert_threshold") {dp_.invert = p.as_bool();}
      else if (n == "blur_ksize") {dp_.blur_ksize = static_cast<int>(p.as_int());}
      else if (n == "open_px") {dp_.open_px = static_cast<int>(p.as_int());}
      else if (n == "close_px") {dp_.close_px = static_cast<int>(p.as_int());}
      else if (n == "min_cluster_px") {dp_.min_cluster_px = static_cast<int>(p.as_int());}
      else if (n == "max_cluster_px") {dp_.max_cluster_px = static_cast<int>(p.as_int());}
      else if (n == "row_step") {dp_.row_step = std::max(1, static_cast<int>(p.as_int()));}
      else if (n == "max_jump_m") {dp_.max_jump_m = p.as_double();}
      else if (n == "min_points") {dp_.min_points = static_cast<int>(p.as_int());}
      else if (n == "roi_x_min") {dp_.roi_x_min = p.as_double();}
      else if (n == "roi_x_max") {dp_.roi_x_max = p.as_double();}
      else if (n == "roi_y_half") {dp_.roi_y_half = p.as_double();}
      else if (n == "raw_top_frac") {raw_top_frac_ = p.as_double();}
      else if (n == "raw_bottom_frac") {raw_bottom_frac_ = p.as_double();}
      else if (n == "raw_left_frac") {raw_left_frac_ = p.as_double();}
      else if (n == "raw_right_frac") {raw_right_frac_ = p.as_double();}
    }
    detector_.set_params(dp_);
    rcl_interfaces::msg::SetParametersResult res;
    res.successful = true;
    return res;
  }

  void onImage(sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
  {
    if (!ipm_.ready() || msg->data.empty()) {return;}
    cv::Mat bgr = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (bgr.empty()) {return;}

    // Raw image-space crop: black out outside the rectangle before warping
    // (keeps full-res coords so the homography still maps correctly).
    if (raw_top_frac_ > 0.0 || raw_bottom_frac_ < 1.0 ||
      raw_left_frac_ > 0.0 || raw_right_frac_ < 1.0)
    {
      const int Hh = bgr.rows, Ww = bgr.cols;
      const int t = std::clamp(static_cast<int>(raw_top_frac_ * Hh), 0, Hh);
      const int b = std::clamp(static_cast<int>(raw_bottom_frac_ * Hh), t, Hh);
      const int l = std::clamp(static_cast<int>(raw_left_frac_ * Ww), 0, Ww);
      const int r = std::clamp(static_cast<int>(raw_right_frac_ * Ww), l, Ww);
      cv::Mat cropped = cv::Mat::zeros(bgr.size(), bgr.type());
      if (r > l && b > t) {
        bgr(cv::Rect(l, t, r - l, b - t)).copyTo(cropped(cv::Rect(l, t, r - l, b - t)));
      }
      bgr = cropped;
    }

    const auto t0 = std::chrono::steady_clock::now();
    cv::Mat bird;
    ipm_.warp(bgr, bird);

    const bool want_debug =
      debug_pub_ && debug_pub_->get_subscription_count() > 0;
    lane_pilot::DetectionResult det = detector_.detect(bird, ipm_, want_debug);
    const auto t1 = std::chrono::steady_clock::now();

    geometry_msgs::msg::PoseArray pa;
    pa.header.stamp = msg->header.stamp;
    pa.header.frame_id = base_frame_;
    if (det.valid) {
      pa.poses.reserve(det.points.size());
      for (const auto & cp : det.points) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = cp.x;
        pose.position.y = cp.y;
        pose.position.z = 0.0;
        pose.orientation.w = 1.0;
        pa.poses.push_back(pose);
      }
    }
    points_pub_->publish(pa);

    // ---- stats ----
    ++frame_count_;
    detect_ms_sum_ +=
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    point_sum_ += det.points.size();
    if (det.valid) {++valid_count_;}

    if (want_debug) {publishDebug(bird, det, msg->header.stamp);}
  }

  void publishDebug(
    const cv::Mat & bird, const lane_pilot::DetectionResult & det,
    const builtin_interfaces::msg::Time & stamp)
  {
    cv::Mat viz;
    if (bird.channels() == 1) {
      cv::cvtColor(bird, viz, cv::COLOR_GRAY2BGR);
    } else {
      viz = bird.clone();
    }
    const int W = viz.cols;

    // Y = 0 reference (image center column) and a few forward-distance ticks.
    cv::line(viz, {W / 2, 0}, {W / 2, viz.rows - 1}, cv::Scalar(120, 120, 120), 1);
    for (double xm = 0.1; xm < ipm_.view().x_max; xm += 0.1) {
      double c, r;
      ipm_.ground_to_bird(xm, 0.0, c, r);
      const int rr = static_cast<int>(std::lround(r));
      cv::line(viz, {0, rr}, {W - 1, rr}, cv::Scalar(60, 60, 60), 1);
      char lab[16];
      std::snprintf(lab, sizeof(lab), "%.1f", xm);
      cv::putText(viz, lab, {2, rr - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.3,
        cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
    }

    // ROI box (orange) — the ground band the detector searches.
    double col_l, row_t, col_r, row_b;
    ipm_.ground_to_bird(dp_.roi_x_max, dp_.roi_y_half, col_l, row_t);
    ipm_.ground_to_bird(dp_.roi_x_min, -dp_.roi_y_half, col_r, row_b);
    cv::rectangle(
      viz,
      cv::Point(static_cast<int>(std::lround(col_l)), static_cast<int>(std::lround(row_t))),
      cv::Point(static_cast<int>(std::lround(col_r)), static_cast<int>(std::lround(row_b))),
      cv::Scalar(0, 140, 255), 1);

    for (size_t i = 0; i < det.bird_pts.size(); ++i) {
      cv::circle(viz, det.bird_pts[i], 2, cv::Scalar(0, 255, 0), -1);
      if (i > 0) {
        cv::line(viz, det.bird_pts[i - 1], det.bird_pts[i],
          cv::Scalar(0, 200, 0), 1);
      }
    }

    char buf[128];
    std::snprintf(
      buf, sizeof(buf), "%s n=%zu conf=%.2f",
      det.valid ? "VALID" : "weak", det.points.size(), det.confidence);
    cv::putText(viz, buf, {6, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
      cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(viz, buf, {6, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    debug_pub_->publish(
      bgr_to_compressed_jpeg(viz, stamp, base_frame_, debug_jpeg_quality_));
  }

  void logStats()
  {
    if (frame_count_ == 0) {
      if (!ipm_.ready()) {
        RCLCPP_WARN(get_logger(), "idle: no valid homography loaded.");
      }
      return;
    }
    const double rate = frame_count_ / 5.0;
    RCLCPP_INFO(
      get_logger(),
      "%.1f fps | detect %.1f ms | %.1f pts/frame | %.0f%% valid",
      rate, detect_ms_sum_ / frame_count_,
      static_cast<double>(point_sum_) / frame_count_,
      100.0 * valid_count_ / frame_count_);
    frame_count_ = 0;
    detect_ms_sum_ = 0.0;
    point_sum_ = 0;
    valid_count_ = 0;
  }

  std::string homography_file_;
  std::string base_frame_;
  bool publish_debug_image_;
  int debug_jpeg_quality_;
  double raw_top_frac_, raw_bottom_frac_, raw_left_frac_, raw_right_frac_;

  lane_pilot::Ipm ipm_;
  lane_pilot::LaneDetector detector_;
  lane_pilot::DetectorParams dp_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // stats
  int frame_count_ = 0;
  double detect_ms_sum_ = 0.0;
  size_t point_sum_ = 0;
  int valid_count_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr points_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaneIpmNode>());
  rclcpp::shutdown();
  return 0;
}
