cmake_minimum_required(VERSION 3.8)
project(box_detector)
## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
#add_definitions(-Wall -Werror)

add_definitions(-DCURRENT_PKG_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  /opt/openvino/runtime/include/ie/
  /opt/openvino/runtime/include/ngraph/
  /opt/openvino/runtime/include/openvino/
  include/box_detector/

)
set(CMAKE_PREFIX_PATH  /opt/openvino/runtime)

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(visualization_msgs REQUIRED)


ament_auto_find_build_dependencies()


###########
## Build ##
###########
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_box::BoxDetector
  EXECUTABLE ${PROJECT_NAME}_node
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES})
  

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
  ament_cmake_copyright
  ament_cmake_uncrustify
  ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()
  
  add_definitions(-DTEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  find_package(ament_cmake_gtest)
  set(TEST_NAME test_node_video)
  ament_add_gtest(${TEST_NAME} test/${TEST_NAME}.cpp)
  target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
)