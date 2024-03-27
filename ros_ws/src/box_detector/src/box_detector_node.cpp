#include "box_detector/box_detector_node.hpp"
#include "geometry_msgs/msg/point.hpp"
namespace rm_auto_box
{
  BoxDetector::BoxDetector(const rclcpp::NodeOptions &options)
      : Node("box_detector", options)
  {
    RCLCPP_INFO(this->get_logger(), "Starting BoxDetectorNode!");
    // Detector
    // detector = initDetector();

     debug_ = this->declare_parameter("debug", false);
    if (debug_)
    {
      createDebugPublishers(); // 发布识别结果的image
    }

    // Debug param change moniter
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
        debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter &p)
                                                 {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers(); });
    
    // 接受相机的消息
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
        {
          cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
          cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
          pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
          cam_info_sub_.reset();
        });

    // 相机视频接收者
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&BoxDetector::ImageCallback, this, std::placeholders::_1));
    boxs_pub_ = this->create_publisher<auto_aim_interfaces::msg::Boxs>("/detector/boxs", rclcpp::SensorDataQoS());
  } 

  void BoxDetector::ImageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg)
  {

  }         
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_box::BoxDetector)