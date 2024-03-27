#include <string>
#include "box_detector/box_detector.hpp"
#include "gtest/gtest.h"
#include <rclcpp/executors.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/utilities.hpp>
#include "box_detector/box_detector_node.hpp"
#include "auto_aim_interfaces/msg/tracker2_d.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

using namespace rm_auto_box;

class Test_node : public rclcpp::Node
{
public:
  cv::Point2f pre_Point;
    Test_node(const rclcpp::NodeOptions &options) : Node("TestNode", options)
    {
        timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);
        Test_Sub = this->create_subscription<auto_aim_interfaces::msg::Tracker2D>("/entracker/Target2D", rclcpp::SensorDataQoS(), std::bind(&Test_node::Test_SubCallback, this, std::placeholders::_1));
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        receive_thread_ = std::thread(&Test_node::receiveData, this);
    }
private:
     std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    void receiveData()
    {
        while (rclcpp::ok)
        {
            geometry_msgs::msg::TransformStamped t;
            timestamp_offset_ = get_parameter("timestamp_offset").as_double();
            t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
            t.header.frame_id = "odom";
            t.child_frame_id = "gimbal_link";
            tf2::Quaternion q;
            q.setRPY(0, 0, 0);
            q.setX(0);
            q.setY(0);
            q.setZ(0);
            t.transform.rotation = tf2::toMsg(q);
            tf_broadcaster_->sendTransform(t);
        }
    }
    double timestamp_offset_ = 0.05;
    rclcpp::Subscription<auto_aim_interfaces::msg::Tracker2D>::SharedPtr Test_Sub;
    void Test_SubCallback(const auto_aim_interfaces::msg::Tracker2D::SharedPtr Point)
    {
        RCLCPP_INFO(this->get_logger(), "x:%.2f y:%.2f", Point->x, Point->y);
        pre_Point.x = Point->x;
        pre_Point.y = Point->y;
    }
    std::thread receive_thread_;
 
};


