cmake_minimum_required(VERSION 3.8)
project(auto_aim_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/Armors.msg"
  "msg/Target.msg"

  "msg/DebugLight.msg"
  "msg/DebugLights.msg"
  "msg/DebugArmor.msg"
  "msg/DebugArmors.msg"

  "msg/DebugBox.msg"
  "msg/DebugBoxs.msg"
  "msg/Box.msg"
  "msg/Boxs.msg"
  "msg/BoxTarget.msg"
  
  "msg/TrackerInfo.msg"
  "msg/Leaf.msg"
  "msg/Leafs.msg"
  "msg/DebugLeaf.msg"
  "msg/DebugLeafs.msg"
  "srv/TrackingMode.srv"
  "msg/Tracker2D.msg"
  "msg/EnTarget.msg"

  DEPENDENCIES
    std_msgs
    geometry_msgs
)


ament_package()
