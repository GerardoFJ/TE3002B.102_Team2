#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_ros/transform_broadcaster.h"

class CmdVelToWheels : public rclcpp::Node
{
public:
  CmdVelToWheels()
  : Node("cmd_vel_to_wheels")
  {
    wheel_radius_ = this->declare_parameter<double>("wheel_radius", 0.05225);
    wheel_separation_ = this->declare_parameter<double>("wheel_separation", 0.17387);
    publish_tf_ = this->declare_parameter<bool>("publish_tf", false);
    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");

    left_pub_ = this->create_publisher<std_msgs::msg::Float32>("/VelocitySetL", 10);
    right_pub_ = this->create_publisher<std_msgs::msg::Float32>("/VelocitySetR", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 50);

    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10,
      std::bind(&CmdVelToWheels::cmdVelCallback, this, std::placeholders::_1));

    const auto sensor_qos = rclcpp::SensorDataQoS();
    enc_left_sub_ = this->create_subscription<std_msgs::msg::Float32>(
      "/VelocityEncL", sensor_qos,
      std::bind(&CmdVelToWheels::encLeftCallback, this, std::placeholders::_1));
    enc_right_sub_ = this->create_subscription<std_msgs::msg::Float32>(
      "/VelocityEncR", sensor_qos,
      std::bind(&CmdVelToWheels::encRightCallback, this, std::placeholders::_1));

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    RCLCPP_INFO(
      this->get_logger(),
      "cmd_vel_to_wheels ready (r=%.5f m, L=%.5f m, publish_tf=%s)",
      wheel_radius_, wheel_separation_, publish_tf_ ? "true" : "false");
  }

private:
  void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const double v = msg->linear.x;
    const double w = msg->angular.z;

    const double v_left = (v - w * wheel_separation_ / 2.0) / wheel_radius_;
    const double v_right = (v + w * wheel_separation_ / 2.0) / wheel_radius_;

    std_msgs::msg::Float32 left_msg;
    left_msg.data = static_cast<float>(v_left);
    std_msgs::msg::Float32 right_msg;
    right_msg.data = static_cast<float>(v_right);

    left_pub_->publish(left_msg);
    right_pub_->publish(right_msg);
  }

  void encLeftCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    wL_ = static_cast<double>(msg->data);
    updateOdom();
  }

  void encRightCallback(const std_msgs::msg::Float32::SharedPtr msg)
  {
    wR_ = static_cast<double>(msg->data);
    updateOdom();
  }

  void updateOdom()
  {
    const rclcpp::Time now = this->get_clock()->now();
    if (!last_time_.has_value()) {
      last_time_ = now;
      return;
    }
    const double dt = (now - *last_time_).seconds();
    if (dt <= 0.0) {
      return;
    }
    last_time_ = now;

    const double v = wheel_radius_ * (wR_ + wL_) / 2.0;
    const double w = wheel_radius_ * (wR_ - wL_) / wheel_separation_;

    const double next_theta = theta_ + w * dt;
    theta_ = std::atan2(std::sin(next_theta), std::cos(next_theta));
    x_ += v * std::cos(theta_) * dt;
    y_ += v * std::sin(theta_) * dt;

    const double qz = std::sin(theta_ / 2.0);
    const double qw = std::cos(theta_ / 2.0);
    const auto stamp = now;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = x_;
    odom.pose.pose.position.y = y_;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.z = qz;
    odom.pose.pose.orientation.w = qw;
    odom.twist.twist.linear.x = v;
    odom.twist.twist.angular.z = w;
    odom_pub_->publish(odom);

    if (tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = stamp;
      t.header.frame_id = odom_frame_;
      t.child_frame_id = base_frame_;
      t.transform.translation.x = x_;
      t.transform.translation.y = y_;
      t.transform.translation.z = 0.0;
      t.transform.rotation.z = qz;
      t.transform.rotation.w = qw;
      tf_broadcaster_->sendTransform(t);
    }
  }

  double wheel_radius_;
  double wheel_separation_;
  bool publish_tf_;
  std::string odom_frame_;
  std::string base_frame_;

  double wL_{0.0};
  double wR_{0.0};
  double x_{0.0};
  double y_{0.0};
  double theta_{0.0};
  std::optional<rclcpp::Time> last_time_;

  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr left_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr right_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr enc_left_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr enc_right_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelToWheels>());
  rclcpp::shutdown();
  return 0;
}
