#include "box_detector/box_detector_node.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "opencv2/opencv.hpp"
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

  cv::Mat BoxDetector::VideoTest(cv::Mat &img, cv::Mat &bin)
  {
    boxs_msg_.boxs.clear();
    cv::Mat result_img = img.clone();
    boxs_msg_.header.stamp = this->now();
    auto final_time = this->now();
    auto boxs = detector_->detect(img);
    auto latency = (this->now() - final_time).seconds() * 1000;
    std::stringstream latency_ss;
    latency_ss << "Latency: " << latency << "ms" << std::endl; // 计算图像处理的延迟输出到日志中
    auto latency_s = latency_ss.str();
    std::cout << latency_s << std::endl;
    sum_latency += latency;
    count++;
    detector_->drawRuselt(result_img);
    cv::putText(
        result_img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    return result_img;
  }

  void BoxDetector::ImageCallback(const sensor_msgs::msg::Image::SharedPtr img_msg)
  {
    auto boxs = detectBoxs(img_msg);
    if(pnp_solver_ != nullptr)
    {
      boxs_msg_.header = img_msg->header;
      boxs_msg_.boxs.clear();

      auto_aim_interfaces::msg::Box box_msg;
      for (const auto &box : boxs)
      {
        // prob
        box_msg.prob = box.prob;
        // type
        box_msg.type = box.box_type;
        // debug boxs
        cv::Mat rvec, tvec;
        bool success = pnp_solver_->solvePnP_(box, rvec, tvec);
        if (success)
        {
          RCLCPP_INFO(this->get_logger(), "pnp success");
          // fill pose
          box_msg.pose.position.x = tvec.at<double>(0);
          box_msg.pose.position.y = tvec.at<double>(1);
          box_msg.pose.position.z = tvec.at<double>(2);
          // debug
          if(debug_)
          {
            RCLCPP_INFO(this->get_logger(), "x:%.2f y:%.2f z:%.2f", box_msg.pose.position.x, box_msg.pose.position.y, box_msg.pose.position.z);
          }
          // rvec to 3x3 rotation matrix
          cv::Mat rotation_matrix;
          cv::Rodrigues(rvec, rotation_matrix);
          // rotation matrix to quaternion
          tf2::Matrix3x3 tf2_rotation_matrix(
              rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
              rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
              rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
              rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
              rotation_matrix.at<double>(2, 2));
          tf2::Quaternion tf2_q;
          tf2_rotation_matrix.getRotation(tf2_q);
          box_msg.pose.orientation = tf2::toMsg(tf2_q);

          // fill the target
          boxs_msg_.boxs.emplace_back(box_msg);

        }
        else
        {
          RCLCPP_INFO(this->get_logger(), "pnp failed QwQ!");
        }

      }
      // publishing boxs 
      boxs_pub_->publish(boxs_msg_);
    }
  }

  void BoxDetector::createDebugPublishers()
  {
    boxs_data_pub = this->create_publisher<auto_aim_interfaces::msg::DebugBoxs>("/detector/debug_boxs", 10);
    binary_img_pub_ = image_transport::create_publisher(this, "/detector/box_binary_img");
    result_img_pub_ = image_transport::create_publisher(this, "/detector/box_result_img");
  } 

  void BoxDetector::destroyDebugPublishers()
  {
    boxs_data_pub.reset();
    result_img_pub_.shutdown();
    binary_img_pub_.shutdown();
  }   

  std::unique_ptr<Box_Detector> BoxDetector::initDetector()
  {
    RCLCPP_INFO(this->get_logger(), "<节点初始化> 检测器参数加载中");
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    // conf_threshold
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value = 255;
    int binary_thres = declare_parameter("binary_thres", 100, param_desc);
    param_desc.integer_range[0].step = 0.01;
    param_desc.integer_range[0].from_value = 0.0;
    param_desc.integer_range[0].to_value = 1.0;
    float conf_threshold = declare_parameter("conf_threshold", 0.6, param_desc);
    // NMS_THRESHOLD
    param_desc.integer_range[0].from_value = 0.0;
    param_desc.integer_range[0].to_value = 1.0;
    float nms_threshold = declare_parameter("nms_threshold", 0.1, param_desc);
    // detect color
    param_desc.description = "0-RED, 1-BLUE";
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value = 1;
    auto detect_color = declare_parameter("detect_color", BLUE, param_desc);
    auto detector = std::make_unique<Box_Detector>(nms_threshold, conf_threshold, detect_color, binary_thres);
    return detector;
  }    

  std::vector<Box> BoxDetector::detectBoxs(
    const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
  { // Convert ROS img to cv::Mat
    auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;
    // Update params
    detector_->CONF_THRESHOLD = get_parameter("conf_threshold").as_double();
    detector_->NMS_THRESHOLD = get_parameter("nms_threshold").as_double();
    detector_->detect_color = get_parameter("detect_color").as_int();
    detector_->binary_thres = get_parameter("binary_thres").as_int();
    auto boxs = detector_->detect(img); // detect box

    auto final_time = this->now();
    auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");
    // Publish debug info
    if (debug_)
    {
      
      binary_img_pub_.publish(
          cv_bridge::CvImage(img_msg->header, "mono8", detector_->bin).toImageMsg());
      boxs_data_pub->publish(detector_->debug_boxs);
      cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
      detector_->drawRuselt(img);
      // Draw camera center
      // Draw latency
      std::stringstream latency_ss;
      latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
      auto latency_s = latency_ss.str();
      cv::putText(
          img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
      result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
    }
    return boxs;

  }
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_box::BoxDetector)