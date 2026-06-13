// safety_controller_node — Phase 3 of the ML line follower.
//
// Subscribes to:
//   /navigation/omega          (std_msgs/Float64) — raw ML prediction
//   /traffic_light/go          (std_msgs/Bool, optional) — stop on red
//
// Publishes /cmd_vel at a fixed rate (default 20 Hz) regardless of
// either input's actual rate, so the motor driver never gets starved.
// If /navigation/omega goes silent for `omega_timeout_s`, the controller
// holds zero (safer than coasting a stale heading).
//
// Filter (matches PDF section 16):
//   omega_safe = alpha * omega_prev + (1 - alpha) * omega_pred
//   omega_safe = clip(omega_safe, -omega_max, omega_max)
// Then a linear velocity is set (constant `linear_speed` * traffic_light
// gate * optional slow-on-turn dampening).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"

class SafetyController : public rclcpp::Node
{
public:
  SafetyController()
  : Node("safety_controller_node")
  {
    linear_speed_ = declare_parameter<double>("linear_speed", 0.10);
    omega_max_ = declare_parameter<double>("omega_max", 0.8);
    alpha_ = std::clamp(
      declare_parameter<double>("smooth_alpha", 0.65), 0.0, 0.99);
    slow_on_turn_ = std::clamp(
      declare_parameter<double>("slow_on_turn", 0.3), 0.0, 1.0);
    omega_timeout_s_ = declare_parameter<double>("omega_timeout_s", 0.5);
    publish_rate_ = std::max(
      1.0, declare_parameter<double>("publish_rate", 20.0));
    require_go_ = declare_parameter<bool>("require_traffic_light_go", false);

    omega_sub_ = create_subscription<std_msgs::msg::Float64>(
      "/navigation/omega", 10,
      std::bind(&SafetyController::onOmega, this, std::placeholders::_1));

    go_sub_ = create_subscription<std_msgs::msg::Bool>(
      "/traffic_light/go", 10,
      std::bind(&SafetyController::onGo, this, std::placeholders::_1));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    last_omega_time_ = now();
    last_log_time_ = now();
    cmd_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / publish_rate_)),
      std::bind(&SafetyController::tick, this));
    log_timer_ = create_wall_timer(
      std::chrono::seconds(5),
      std::bind(&SafetyController::logStats, this));

    RCLCPP_INFO(
      get_logger(),
      "safety_controller_node started "
      "(lin=%.2f omega_max=%.2f alpha=%.2f rate=%.0f Hz "
      "timeout=%.2fs require_go=%s).",
      linear_speed_, omega_max_, alpha_, publish_rate_, omega_timeout_s_,
      require_go_ ? "true" : "false");
  }

private:
  void onOmega(std_msgs::msg::Float64::ConstSharedPtr msg)
  {
    latest_omega_ = msg->data;
    last_omega_time_ = now();
    ++omega_count_;
  }

  void onGo(std_msgs::msg::Bool::ConstSharedPtr msg)
  {
    traffic_light_go_ = msg->data;
    have_go_ = true;
  }

  void tick()
  {
    const auto t_now = now();
    const double since_omega = (t_now - last_omega_time_).seconds();

    geometry_msgs::msg::Twist out;

    // Stale input safety: if we haven't heard a fresh /navigation/omega
    // recently, hold zero. Better to stand still than carry a stale
    // heading off the track.
    if (!latest_omega_.has_value() || since_omega > omega_timeout_s_) {
      omega_smooth_ = 0.0;
      out.linear.x = 0.0;
      out.angular.z = 0.0;
      cmd_pub_->publish(out);
      stale_ticks_++;
      return;
    }

    // Exponential smoothing — matches PDF safety filter.
    omega_smooth_ = alpha_ * omega_smooth_ + (1.0 - alpha_) * (*latest_omega_);
    omega_smooth_ = std::clamp(omega_smooth_, -omega_max_, omega_max_);

    // Optional traffic-light gate. If the param requires it and we
    // haven't heard a Bool yet, default to stopped (safer than rolling
    // through the start line).
    bool go = true;
    if (require_go_) {
      go = have_go_ && traffic_light_go_;
    } else if (have_go_) {
      go = traffic_light_go_;  // honour the topic if it's around
    }

    double lin = linear_speed_;
    if (!go) {
      lin = 0.0;
    } else if (slow_on_turn_ > 0.0) {
      // Pull down linear when omega is large, like line_follower used to.
      const double frac = std::abs(omega_smooth_) / omega_max_;
      lin = linear_speed_ * (1.0 - slow_on_turn_ * std::clamp(frac, 0.0, 1.0));
      lin = std::max(0.0, lin);
    }

    out.linear.x = lin;
    out.angular.z = omega_smooth_;
    cmd_pub_->publish(out);
  }

  void logStats()
  {
    const auto t = now();
    const double dt = (t - last_log_time_).seconds();
    if (dt > 0.0) {
      RCLCPP_INFO(
        get_logger(),
        "omega_in=%.1f Hz  omega_smooth=%+.3f  stale_ticks=%lu",
        omega_count_ / dt, omega_smooth_,
        static_cast<unsigned long>(stale_ticks_));
    }
    omega_count_ = 0;
    stale_ticks_ = 0;
    last_log_time_ = t;
  }

  // Parameters
  double linear_speed_;
  double omega_max_;
  double alpha_;
  double slow_on_turn_;
  double omega_timeout_s_;
  double publish_rate_;
  bool require_go_;

  // State
  std::optional<double> latest_omega_;
  rclcpp::Time last_omega_time_;
  double omega_smooth_ = 0.0;
  bool have_go_ = false;
  bool traffic_light_go_ = true;
  uint64_t omega_count_ = 0;
  uint64_t stale_ticks_ = 0;
  rclcpp::Time last_log_time_;

  // ROS
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr omega_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr go_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr cmd_timer_;
  rclcpp::TimerBase::SharedPtr log_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SafetyController>());
  rclcpp::shutdown();
  return 0;
}
