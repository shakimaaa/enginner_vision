#include "box_detector.hpp"

namespace rm_auto_box
{
  std::vector<Box> Box_Detector::detect(const cv::Mat &input)
  {
    result_img = input.clone();
    auto boxs = work(result_img);
    boxs_ = Box_filter(boxs, input.cols, input.rows);
    // Box_Points = findBox(input);
    auto_aim_interfaces::msg::DebugBox debug_box;
    for (auto &box : boxs)
    {
      debug_box.label=box.label;
      // debug_box.r_center.x=R_Point.x;
      // debug_box.r_center.y=R_Point.y;
      debug_box.point_0.x=box.kpt[BoxPointType::POINT_0].x;
      debug_box.point_0.y=box.kpt[BoxPointType::POINT_0].y;
      debug_box.point_1.x=box.kpt[BoxPointType::POINT_1].x;
      debug_box.point_1.y=box.kpt[BoxPointType::POINT_1].y;
      debug_box.point_2.x=box.kpt[BoxPointType::POINT_2].x;
      debug_box.point_2.y=box.kpt[BoxPointType::POINT_2].y;
      debug_box.point_3.x=box.kpt[BoxPointType::POINT_3].x;
      debug_box.point_3.y=box.kpt[BoxPointType::POINT_3].y;
      debug_box.box_center.x=box.kpt[BoxPointType::CENTER_POINT].x;
      debug_box.box_center.y=box.kpt[BoxPointType::CENTER_POINT].y;
    }

    return boxs_;
  }

  Box_Detector::Box_Detector(float NMS_THRESHOLD, float CONF_THRESHOLD, int detect_color, int binary_thres) 
      : NMS_THRESHOLD(NMS_THRESHOLD), CONF_THRESHOLD(CONF_THRESHOLD), detect_color(detect_color), binary_thres(binary_thres)
  {
    model = core.read_model(MODEL_PATH);
    compiled_model = core.compile_model(model, DEVICE);
    infer_request = compiled_model.create_infer_request();
    input_port = compiled_model.input();
    kernel_3_3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    kernel_5_5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    kernel_7_7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
  }  

  cv::Mat Box_Detector::letter_box(cv::Mat &src)
  {
    int col = src.cols;
    int row = src.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    src.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  std::vector<Box> Box_Detector::work(cv::Mat src_img)
  {
    this->debug_boxs.data.clear();
    this->boxs_.clear();
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    // Preprocess the image
    cv::Mat boxed = letter_box(src_img);
    float scale = boxed.size[0] / IMG_SIZE;
    cv::Mat blob = cv::dnn::blobFromImage(boxed, 1.0 / 255.0, cv::Size(IMG_SIZE, IMG_SIZE), cv::Scalar(), true);
    //  Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    //  Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);
    // infer_request.infer();
    infer_request.start_async();
    infer_request.wait();
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    // Postprocess the result
    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    cv::transpose(output_buffer, output_buffer); //[8400,13]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> objects_keypoints;
     // box[cx, cy, w, h] + Score + [4,2] keypoints
    for (int i = 0; i < output_buffer.rows; i++)
    {
      float class_score = output_buffer.at<float>(i, 4);
      if (class_score < CONF_THRESHOLD)
        continue;
      else
      {
        class_scores.emplace_back(class_score);
        class_ids.emplace_back(0); //{0:"leaf"}
        float cx = output_buffer.at<float>(i, 0);
        float cy = output_buffer.at<float>(i, 1);
        float w = output_buffer.at<float>(i, 2);
        float h = output_buffer.at<float>(i, 3);

        // Get the box
        int left = int((cx - 0.5 * w) * scale);
        int top = int((cy - 0.5 * h) * scale);
        int width = int(w * scale);
        int height = int(h * scale);

        // Get the keypoints
        std::vector<cv::Point2f> keypoints;
        cv::Mat kpts = output_buffer.row(i).colRange(5, 13);
        for (int j = 0; j < KPT_NUM; j++)
        {
          float x = kpts.at<float>(0, j * 2 + 0) * scale;
          float y = kpts.at<float>(0, j * 2 + 1) * scale;
          cv::Point2f kpt(x, y);
          keypoints.emplace_back(kpt);
        }
        boxes.emplace_back(cv::Rect(left, top, width, height));
        objects_keypoints.emplace_back(keypoints);
      }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
    
    // -------- Select the detection best result -----------
    for (size_t i = 0; i < indices.size(); i++)
    {
      int index = indices[i]; // best result index
      Box box;
      box.rect = boxes[index];
      box.label = class_ids[index];
      box.prob = class_scores[index];

      std::vector<cv::Point2f> object_keypoints = objects_keypoints[index];
      for (int i = 0; i < KPT_NUM; i++)
      {
        int x = std::clamp(int(object_keypoints[i].x), 0, src_img.cols);
        int y = std::clamp(int(object_keypoints[i].y), 0, src_img.rows);
        // Draw point
        box.kpt.emplace_back(cv::Point2f(x, y));
      }
      boxs_.emplace_back(box);
    }
    return boxs_;
  }
  
  // matching the fourth keypoint for PnP
  std::vector<Box> Box_Detector::Box_filter(
        std::vector<Box> &boxs, const int MAX_WIDTH, const int MAX_HEIGHT)
  {
    auto Get_point = [&](cv::Point2f pt1, cv::Point2f pt2) -> cv::Point2f
    {
      return cv::Point2f((pt1.x + pt2.x) / 2.0, (pt1.y + pt2.y) / 2.0);
    };

    std::vector<Box> result;
    for (auto &box : boxs)
    {
      cv::Point2f last_kpoint(0, 0);
      std::vector<bool> valid_keypoints(3, false);
      for (int i = 0; i < box.kpt.size(); i++)
      {
        if(box.kpt[i].x != 0 && box.kpt[i].y != 0)
        {
          valid_keypoints[i] = true;
        }
      }

      if (valid_keypoints[0] && valid_keypoints[1] && valid_keypoints[2])
      {
        last_kpoint = Get_point(box.kpt[1], box.kpt[2]);
      }
      box.kpt.emplace_back(last_kpoint);
      box.box_type = true;
      for(size_t i = 0; i < box.kpt.size(); i++)
      {
        if (box.kpt[i].x < 0 or box.kpt[i].x > MAX_WIDTH or box.kpt[i].y < 0 or
            box.kpt[i].x > MAX_HEIGHT or box.kpt[i].x < 0 or box.kpt[i].x > MAX_WIDTH or
            box.kpt[i].y < 0 or box.kpt[i].y > MAX_HEIGHT )
        {
          box.box_type = false;
          break;
        }    
      }
      result.emplace_back(box);
    }

    return result;
  } 

  void Box_Detector::drawRuselt(cv::Mat &src)
  {
    for (auto &box : boxs_)
    {
      float x0 = box.rect.x;
      float y0 = box.rect.y;
      float x1 = box.rect.x + box.rect.width;
      float y1 = box.rect.y + box.rect.height;
      int baseLine;
      float prob = box.prob;
      cv::rectangle(src, box.rect, cv::Scalar(0, 0, 255), 2, 8);
      std::string label = class_names[box.label] + std::to_string(box.prob).substr(0, 4);
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
      cv::Rect textBox(box.rect.tl().x, box.rect.tl().y - 15, textSize.width, textSize.height + 5);
      cv::rectangle(src, textBox, cv::Scalar(0, 0, 255), cv::FILLED);
      cv::putText(src, label, cv::Point(box.rect.tl().x, box.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
       for (auto point : box.kpt)
      {
        cv::circle(src, point, 5, cv::Scalar(255, 0, 255), -1);
      }
    }
  } 

  // std::vector<cv::Point2f> Box_Detector::findBox(cv::Mat src)
  // {

  // }


} // namespace rm_auto_box