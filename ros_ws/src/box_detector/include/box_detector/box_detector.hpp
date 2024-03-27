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
#define KPT_NUM 4
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
  };
}