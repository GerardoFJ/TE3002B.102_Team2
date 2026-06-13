// lane_pilot_node — control half of the metric lane follower.
//
// Subscribes the metric centerline points from lane_ipm_node and the robot's
// wheel odometry, and:
//   1. transforms each centerline point into the odom frame and stores it in a
//      rolling memory buffer. Because the buffer is world-fixed, points seen a
//      moment ago at +5 cm are remembered after the robot drives over them —
//      this is what fills the camera blind spot directly in front of the robot.
//   2. on a fixed-rate timer, transforms the buffer back into the *current*
//      base_link frame, fits a smooth centerline polynomial (covering the blind
//      spot from memory), and runs regulated pure pursuit on it.
//
// Because the IPM blind spot is large (~21 cm on this robot), the lookahead is
// deliberately kept beyond it so the pursuit target lands on REAL perceived
// centerline, not on the extrapolated near field.
//
// Publishes a self-contained top-down debug image on /lane/pilot_debug/compressed
// (memory points, perceived points, fitted centerline, lookahead, live v/omega)
// so the controller's worldview can be watched live in rqt_image_view.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/bool.hpp"

namespace
{

struct Pose2D
{
  double t;        // seconds
  double x;
  double y;
  double theta;
};

struct MemPoint
{
  double ox;       // odom frame
  double oy;
  double t;        // seconds (capture time)
};

double yaw_from_quat(double z, double w)
{
  return std::atan2(2.0 * w * z, 1.0 - 2.0 * z * z);
}

double ang_lerp(double a, double b, double f)
{
  double d = std::atan2(std::sin(b - a), std::cos(b - a));
  return a + f * d;
}

// Solve A x = b (A row-major n*n) with Gaussian elimination + partial pivot.
bool solve_linear(int n, std::vector<double> A, std::vector<double> b,
  std::vector<double> & x)
{
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::abs(A[col * n + col]);
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(A[r * n + col]);
      if (v > best) {best = v; piv = r;}
    }
    if (best < 1e-12) {return false;}
    if (piv != col) {
      for (int c = 0; c < n; ++c) {std::swap(A[piv * n + c], A[col * n + c]);}
      std::swap(b[piv], b[col]);
    }
    const double d = A[col * n + col];
    for (int r = col + 1; r < n; ++r) {
      const double f = A[r * n + col] / d;
      if (f == 0.0) {continue;}
      for (int c = col; c < n; ++c) {A[r * n + c] -= f * A[col * n + c];}
      b[r] -= f * b[col];
    }
  }
  x.assign(n, 0.0);
  for (int row = n - 1; row >= 0; --row) {
    double s = b[row];
    for (int c = row + 1; c < n; ++c) {s -= A[row * n + c] * x[c];}
    x[row] = s / A[row * n + row];
  }
  return true;
}

// Least-squares polynomial fit, y = sum_k coeffs[k] x^k.
std::vector<double> fit_poly(
  const std::vector<double> & xs, const std::vector<double> & ys, int deg)
{
  const int m = deg + 1;
  std::vector<double> A(m * m, 0.0), b(m, 0.0);
  std::vector<double> p(2 * deg + 1);
  for (size_t k = 0; k < xs.size(); ++k) {
    p[0] = 1.0;
    for (int i = 1; i <= 2 * deg; ++i) {p[i] = p[i - 1] * xs[k];}
    for (int i = 0; i < m; ++i) {
      b[i] += p[i] * ys[k];
      for (int j = 0; j < m; ++j) {A[i * m + j] += p[i + j];}
    }
  }
  std::vector<double> coeffs;
  if (!solve_linear(m, A, b, coeffs)) {coeffs.assign(m, 0.0);}
  return coeffs;
}

double eval_poly(const std::vector<double> & c, double x)
{
  double y = 0.0;
  for (size_t i = c.size(); i-- > 0; ) {y = y * x + c[i];}
  return y;
}

}  // namespace

class LanePilotNode : public rclcpp::Node
{
public:
  LanePilotNode()
  : Node("lane_pilot_node")
  {
    v_max_ = declare_parameter<double>("v_max", 0.12);
    v_min_ = declare_parameter<double>("v_min", 0.03);
    max_angular_ = declare_parameter<double>("max_angular", 1.8);

    // Lookahead kept beyond the ~21 cm blind spot so the pursuit target is on
    // real perceived centerline, not the extrapolated near field.
    ld_base_ = declare_parameter<double>("lookahead_base", 0.30);
    ld_k_ = declare_parameter<double>("lookahead_k", 0.2);
    ld_min_ = declare_parameter<double>("lookahead_min", 0.25);
    ld_max_ = declare_parameter<double>("lookahead_max", 0.45);

    curv_slow_gain_ = declare_parameter<double>("curv_slow_gain", 0.7);
    curv_kappa_ref_ = declare_parameter<double>("curv_kappa_ref", 4.0);
    curv_eval_x_ = declare_parameter<double>("curv_eval_x", 0.30);

    max_degree_ = std::clamp(static_cast<int>(declare_parameter<int>("max_degree", 2)), 1, 3);

    mem_time_ = declare_parameter<double>("mem_time", 3.0);
    mem_max_points_ = declare_parameter<int>("mem_max_points", 1500);
    x_keep_min_ = declare_parameter<double>("x_keep_min", -0.20);
    x_keep_max_ = declare_parameter<double>("x_keep_max", 0.80);
    y_keep_ = declare_parameter<double>("y_keep", 0.40);

    min_points_for_fit_ = declare_parameter<int>("min_points_for_fit", 5);
    min_span_ = declare_parameter<double>("min_span", 0.05);
    short_path_x_ = declare_parameter<double>("short_path_x", 0.30);
    lost_timeout_ = declare_parameter<double>("lost_timeout", 0.6);

    enable_traffic_light_ = declare_parameter<bool>("enable_traffic_light", true);
    const std::string go_topic = declare_parameter<std::string>(
      "traffic_light_topic", "/traffic_light/go");

    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    const std::string odom_topic = declare_parameter<std::string>(
      "odom_topic", "/odom");
    const std::string points_topic = declare_parameter<std::string>(
      "input_topic", "/lane/points");
    const std::string cmd_topic = declare_parameter<std::string>(
      "cmd_topic", "/cmd_vel");
    publish_path_ = declare_parameter<bool>("publish_path", true);
    publish_debug_image_ = declare_parameter<bool>("publish_debug_image", true);
    debug_jpeg_quality_ = std::clamp(
      static_cast<int>(declare_parameter<int>("debug_jpeg_quality", 70)), 1, 100);
    const double control_rate = std::max(
      1.0, declare_parameter<double>("control_rate", 30.0));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic, 10);
    if (publish_path_) {
      path_pub_ = create_publisher<nav_msgs::msg::Path>("/lane/path", 5);
    }
    if (publish_debug_image_) {
      debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "/lane/pilot_debug/compressed", 5);
    }

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, rclcpp::SensorDataQoS().keep_last(50),
      std::bind(&LanePilotNode::onOdom, this, std::placeholders::_1));
    points_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      points_topic, 10,
      std::bind(&LanePilotNode::onPoints, this, std::placeholders::_1));
    if (enable_traffic_light_) {
      go_sub_ = create_subscription<std_msgs::msg::Bool>(
        go_topic, 10, std::bind(&LanePilotNode::onGo, this, std::placeholders::_1));
    }

    last_points_time_ = now();
    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / control_rate)),
      std::bind(&LanePilotNode::control, this));

    RCLCPP_INFO(
      get_logger(),
      "lane_pilot_node started (v=[%.2f,%.2f] ang_max=%.2f "
      "lookahead=[%.2f,%.2f] mem=%.1fs @ %.0f Hz).",
      v_min_, v_max_, max_angular_, ld_min_, ld_max_, mem_time_, control_rate);
  }

private:
  void onOdom(nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    Pose2D p;
    p.t = rclcpp::Time(msg->header.stamp).seconds();
    p.x = msg->pose.pose.position.x;
    p.y = msg->pose.pose.position.y;
    p.theta = yaw_from_quat(
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    odom_hist_.push_back(p);
    const double cutoff = p.t - (mem_time_ + 1.0);
    while (odom_hist_.size() > 2 && odom_hist_.front().t < cutoff) {
      odom_hist_.pop_front();
    }
    have_odom_ = true;
  }

  std::optional<Pose2D> poseAt(double t) const
  {
    if (odom_hist_.empty()) {return std::nullopt;}
    if (t <= odom_hist_.front().t) {return odom_hist_.front();}
    if (t >= odom_hist_.back().t) {return odom_hist_.back();}
    for (size_t i = 1; i < odom_hist_.size(); ++i) {
      const Pose2D & a = odom_hist_[i - 1];
      const Pose2D & b = odom_hist_[i];
      if (t >= a.t && t <= b.t) {
        const double f = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0.0;
        Pose2D r;
        r.t = t;
        r.x = a.x + f * (b.x - a.x);
        r.y = a.y + f * (b.y - a.y);
        r.theta = ang_lerp(a.theta, b.theta, f);
        return r;
      }
    }
    return odom_hist_.back();
  }

  void onPoints(geometry_msgs::msg::PoseArray::ConstSharedPtr msg)
  {
    if (msg->poses.empty()) {return;}
    const double t = rclcpp::Time(msg->header.stamp).seconds();
    const auto pose = poseAt(t);
    if (!pose) {return;}
    const double ct = std::cos(pose->theta);
    const double st = std::sin(pose->theta);
    perc_pts_.clear();
    for (const auto & ps : msg->poses) {
      const double X = ps.position.x;
      const double Y = ps.position.y;
      perc_pts_.emplace_back(X, Y);                 // base_link, for debug draw
      MemPoint mp;
      mp.ox = pose->x + X * ct - Y * st;
      mp.oy = pose->y + X * st + Y * ct;
      mp.t = t;
      mem_.push_back(mp);
    }
    const double cutoff = t - mem_time_;
    while (!mem_.empty() && mem_.front().t < cutoff) {mem_.pop_front();}
    while (static_cast<int>(mem_.size()) > mem_max_points_) {mem_.pop_front();}
    last_points_time_ = now();
  }

  void onGo(std_msgs::msg::Bool::ConstSharedPtr msg)
  {
    traffic_light_go_ = msg->data;
    have_go_ = true;
  }

  void control()
  {
    if (!have_odom_ || odom_hist_.empty()) {
      publishCmd(0.0, 0.0);
      maybeDrawDebug();
      return;
    }
    const Pose2D robot = odom_hist_.back();
    const double ct = std::cos(robot.theta);
    const double st = std::sin(robot.theta);

    // Memory -> current base_link.
    std::vector<double> xs, ys;
    xs.reserve(mem_.size());
    ys.reserve(mem_.size());
    double max_x = -1e9, min_x = 1e9;
    for (const auto & mp : mem_) {
      const double dx = mp.ox - robot.x;
      const double dy = mp.oy - robot.y;
      const double X = dx * ct + dy * st;
      const double Y = -dx * st + dy * ct;
      if (X < x_keep_min_ || X > x_keep_max_ || std::abs(Y) > y_keep_) {continue;}
      xs.push_back(X);
      ys.push_back(Y);
      max_x = std::max(max_x, X);
      min_x = std::min(min_x, X);
    }

    const bool lost = (now() - last_points_time_).seconds() > lost_timeout_;
    const double span = (xs.empty()) ? 0.0 : (max_x - min_x);
    const bool have_path =
      static_cast<int>(xs.size()) >= min_points_for_fit_ &&
      span >= min_span_ && max_x > 0.02;

    // snapshot for debug
    dbg_.mem_x = xs; dbg_.mem_y = ys; dbg_.max_x = max_x; dbg_.min_x = min_x;
    dbg_.have_path = have_path; dbg_.lost = lost;

    if (!have_path) {
      if (lost) {
        last_v_ *= 0.6;
        last_omega_ *= 0.6;
        if (last_v_ < 0.005) {last_v_ = 0.0; last_omega_ = 0.0;}
      } else {
        last_v_ = std::min(last_v_, v_min_);
        last_omega_ *= 0.8;
      }
      dbg_.coeffs.clear();
      dbg_.v = last_v_; dbg_.omega = last_omega_;
      publishCmd(last_v_, last_omega_);
      maybeDrawDebug();
      return;
    }

    int deg = max_degree_;
    if (static_cast<int>(xs.size()) < deg + 2) {deg = static_cast<int>(xs.size()) - 1;}
    if (span < 0.10) {deg = std::min(deg, 1);} else if (span < 0.25) {deg = std::min(deg, 2);}
    deg = std::clamp(deg, 1, 3);
    const std::vector<double> coeffs = fit_poly(xs, ys, deg);

    const double a1 = coeffs.size() > 1 ? coeffs[1] : 0.0;
    const double a2 = coeffs.size() > 2 ? coeffs[2] : 0.0;
    const double a3 = coeffs.size() > 3 ? coeffs[3] : 0.0;
    const double xe = std::clamp(curv_eval_x_, 0.0, max_x);
    const double dp = a1 + 2.0 * a2 * xe + 3.0 * a3 * xe * xe;
    const double ddp = 2.0 * a2 + 6.0 * a3 * xe;
    const double kappa = std::abs(ddp) / std::pow(1.0 + dp * dp, 1.5);

    double v_cmd = v_max_ *
      (1.0 - curv_slow_gain_ * std::min(1.0, kappa / std::max(1e-6, curv_kappa_ref_)));
    if (max_x < short_path_x_) {
      v_cmd *= std::clamp(max_x / std::max(1e-6, short_path_x_), 0.3, 1.0);
    }
    v_cmd = std::clamp(v_cmd, v_min_, v_max_);

    double ld = std::clamp(ld_base_ + ld_k_ * v_cmd, ld_min_, ld_max_);
    ld = std::min(ld, std::max(ld_min_, max_x));

    double xL = 0.0, yL = 0.0, chordL = 0.0;
    for (double x = 0.0; x <= max_x + 1e-6; x += 0.01) {
      const double y = eval_poly(coeffs, x);
      const double chord = std::hypot(x, y);
      xL = x; yL = y; chordL = chord;
      if (chord >= ld) {break;}
    }

    double omega = 0.0;
    if (chordL > 1e-3) {
      const double gamma = 2.0 * yL / (chordL * chordL);
      omega = std::clamp(v_cmd * gamma, -max_angular_, max_angular_);
    }

    last_v_ = v_cmd;
    last_omega_ = omega;
    dbg_.coeffs = coeffs; dbg_.xL = xL; dbg_.yL = yL; dbg_.ld = ld;
    dbg_.kappa = kappa; dbg_.v = v_cmd; dbg_.omega = omega;

    publishCmd(v_cmd, omega);
    if (publish_path_) {publishPath(coeffs, max_x, robot);}
    maybeDrawDebug();
  }

  void publishCmd(double v, double omega)
  {
    geometry_msgs::msg::Twist cmd;
    dbg_.gate_stop = enable_traffic_light_ && have_go_ && !traffic_light_go_;
    if (!dbg_.gate_stop) {
      cmd.linear.x = v;
      cmd.angular.z = omega;
    }
    cmd_pub_->publish(cmd);
  }

  void publishPath(
    const std::vector<double> & coeffs, double max_x, const Pose2D & robot)
  {
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = odom_frame_;
    const double ct = std::cos(robot.theta);
    const double st = std::sin(robot.theta);
    for (double x = 0.0; x <= max_x + 1e-6; x += 0.02) {
      const double y = eval_poly(coeffs, x);
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = robot.x + x * ct - y * st;
      ps.pose.position.y = robot.y + x * st + y * ct;
      ps.pose.orientation.w = 1.0;
      path.poses.push_back(ps);
    }
    path_pub_->publish(path);
  }

  void maybeDrawDebug()
  {
    if (!debug_pub_ || debug_pub_->get_subscription_count() == 0) {return;}

    // Top-down metric canvas: X forward (up), Y +left (left). 400 px/m.
    const double X_FWD = 0.80, X_BACK = -0.25, Y_MAX = 0.45, ppm = 400.0;
    const int Wd = static_cast<int>((2 * Y_MAX) * ppm);   // 360
    const int Hd = static_cast<int>((X_FWD - X_BACK) * ppm);  // 420
    cv::Mat img(Hd, Wd, CV_8UC3, cv::Scalar(35, 35, 35));
    auto toPix = [&](double X, double Y) {
      return cv::Point(
        static_cast<int>((Y_MAX - Y) * ppm),
        static_cast<int>((X_FWD - X) * ppm));
    };

    // grid: forward-distance lines + labels, Y=0 centerline
    for (double xm = X_BACK; xm <= X_FWD + 1e-6; xm += 0.1) {
      const int rr = toPix(xm, 0.0).y;
      cv::line(img, {0, rr}, {Wd - 1, rr}, cv::Scalar(60, 60, 60), 1);
      char lab[16]; std::snprintf(lab, sizeof(lab), "%.1f", xm);
      cv::putText(img, lab, {2, rr - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.3,
        cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
    }
    cv::line(img, toPix(X_FWD, 0.0), toPix(X_BACK, 0.0),
      cv::Scalar(90, 90, 90), 1);
    // blind-spot band (no camera data below ~0.21 m)
    cv::line(img, toPix(0.21, -Y_MAX), toPix(0.21, Y_MAX),
      cv::Scalar(40, 40, 110), 1);

    // memory points (gray), perceived points (green)
    for (size_t i = 0; i < dbg_.mem_x.size(); ++i) {
      cv::circle(img, toPix(dbg_.mem_x[i], dbg_.mem_y[i]), 1,
        cv::Scalar(150, 150, 150), -1);
    }
    for (const auto & p : perc_pts_) {
      cv::circle(img, toPix(p.first, p.second), 2, cv::Scalar(0, 220, 0), -1);
    }

    // fitted centerline (cyan)
    if (!dbg_.coeffs.empty()) {
      cv::Point prev;
      bool have_prev = false;
      for (double x = 0.0; x <= dbg_.max_x + 1e-6; x += 0.02) {
        const cv::Point pt = toPix(x, eval_poly(dbg_.coeffs, x));
        if (have_prev) {cv::line(img, prev, pt, cv::Scalar(255, 220, 0), 2);}
        prev = pt; have_prev = true;
      }
      // lookahead point + ray from robot
      cv::line(img, toPix(0.0, 0.0), toPix(dbg_.xL, dbg_.yL),
        cv::Scalar(200, 0, 200), 1);
      cv::circle(img, toPix(dbg_.xL, dbg_.yL), 6, cv::Scalar(255, 0, 255), 2);
    }

    // robot (triangle pointing forward) at origin
    const cv::Point o = toPix(0.0, 0.0);
    std::vector<cv::Point> tri{
      {o.x, o.y - 10}, {o.x - 7, o.y + 7}, {o.x + 7, o.y + 7}};
    cv::polylines(img, tri, true, cv::Scalar(0, 165, 255), 2);

    // HUD
    char buf[160];
    const char * status = dbg_.gate_stop ? "TL-STOP"
      : (!dbg_.have_path ? (dbg_.lost ? "LOST" : "NO-PATH") : "TRACK");
    std::snprintf(buf, sizeof(buf), "%s v=%.2f w=%+.2f Ld=%.2f k=%.1f n=%zu",
      status, dbg_.v, dbg_.omega, dbg_.ld, dbg_.kappa, dbg_.mem_x.size());
    cv::putText(img, buf, {6, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
      cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(img, buf, {6, 16}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
      cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp = now();
    msg.header.frame_id = "base_link";
    msg.format = "jpeg";
    std::vector<int> p{cv::IMWRITE_JPEG_QUALITY, debug_jpeg_quality_};
    cv::imencode(".jpg", img, msg.data, p);
    debug_pub_->publish(msg);
  }

  // params
  double v_max_, v_min_, max_angular_;
  double ld_base_, ld_k_, ld_min_, ld_max_;
  double curv_slow_gain_, curv_kappa_ref_, curv_eval_x_;
  int max_degree_;
  double mem_time_;
  int mem_max_points_;
  double x_keep_min_, x_keep_max_, y_keep_;
  int min_points_for_fit_;
  double min_span_, short_path_x_, lost_timeout_;
  bool enable_traffic_light_;
  std::string odom_frame_;
  bool publish_path_;
  bool publish_debug_image_;
  int debug_jpeg_quality_;

  // state
  std::deque<Pose2D> odom_hist_;
  std::deque<MemPoint> mem_;
  std::vector<std::pair<double, double>> perc_pts_;   // latest perception (base_link)
  bool have_odom_ = false;
  rclcpp::Time last_points_time_;
  double last_v_ = 0.0;
  double last_omega_ = 0.0;
  bool have_go_ = false;
  bool traffic_light_go_ = true;

  struct DbgState
  {
    std::vector<double> mem_x, mem_y;
    std::vector<double> coeffs;
    double max_x = 0, min_x = 0, xL = 0, yL = 0, ld = 0, kappa = 0, v = 0, omega = 0;
    bool have_path = false, lost = false, gate_stop = false;
  } dbg_;

  // ROS
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr points_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr go_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LanePilotNode>());
  rclcpp::shutdown();
  return 0;
}
