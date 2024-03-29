#ifndef BOX_DETECTOR_HPP_
#define BOX_DETECTOR_HPP_

#include <sys/types.h>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "box.hpp"
#include "auto_aim_interfaces/msg/debug_boxs.hpp"
using namespace InferenceEngine;

#define IMG_SIZE 416.0      // 推理图像大小
#define DEVICE "CPU"         // 设备选择
#define KPT_NUM 3
#define CLS_NUM 1
#ifdef CURRENT_PKG_DIR
#define MODEL_PATH CURRENT_PKG_DIR "/models/best.onnx"
#endif

namespace rm_auto_box
{
  class Box_Detector
  {
   public:
     Box_Detector(float NMS_THRESHOLD,float CONF_THRESHOLD,int color,int binary_thres);
     std::vector<Box> detect(const cv::Mat &input);
     cv::Mat letter_box(cv::Mat &src);
     std::vector<Box> work(cv::Mat src_img);
     std::vector<Box> Box_filter(
        std::vector<Box> &boxs, const int MAX_WIDTH, const int MAX_HEIGHT);
     void drawRuselt(cv::Mat &src); 
     // std::vector<cv::Point2f> findBox(cv::Mat src);  

     // Debug msgs
    cv::Mat result_img;
    auto_aim_interfaces::msg::DebugBoxs debug_boxs;
    std::vector<Box> boxs_;
    // params
    float NMS_THRESHOLD; // NMS参数
    float CONF_THRESHOLD; // 置信度参数
    int detect_color;
    cv::Mat bin;
    int binary_thres;
    std::vector<cv::Point2f> Box_Points;
   private:
    cv::Mat dilate_struct;
    cv::Mat erode_struct;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node>input_port;
    const std::vector<std::string> class_names = {"box"};
    cv::Mat kernel_3_3,kernel_5_5,kernel_7_7;
  };
} // namespace rm_auto_box
#endif // BOX_DETECTOR_HPP_