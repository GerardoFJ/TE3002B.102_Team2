#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <camera_info_manager/camera_info_manager.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

using namespace std::chrono_literals;

class CameraInfoPublisher : public rclcpp::Node
{
  public:
    CameraInfoPublisher()
    : Node("camera_info_publisher")
    {
      camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 10);
      timer_ = this->create_wall_timer(100ms, std::bind(&CameraInfoPublisher::timer_callback, this));
      
      cinfo_manager_ = std::make_shared<camera_info_manager::CameraInfoManager>(this);
      
      frame_id_ = this->declare_parameter("frame_id", "camera");
      
      auto camera_calibration_file_param_ = this->declare_parameter("camera_calibration_file", "file://config/camera.yaml");
      cinfo_manager_->loadCameraInfo(camera_calibration_file_param_);
    }

  private:
    void timer_callback()
    {
      auto camera_info_msg_ = cinfo_manager_->getCameraInfo();
      
      //sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg_(
      //      new sensor_msgs::msg::CameraInfo(cinfo_manager_->getCameraInfo()));
 
      camera_info_msg_.header.stamp = this->get_clock()->now();
      camera_info_msg_.header.frame_id = frame_id_;

      camera_info_pub_->publish(camera_info_msg_);
    }
    
    std::string frame_id_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_manager_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraInfoPublisher>());
  rclcpp::shutdown();
  return 0;
}

