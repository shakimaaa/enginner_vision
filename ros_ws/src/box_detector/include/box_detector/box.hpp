#ifndef BOX_HPP
#define BOX_HPP


#include <opencv2/core.hpp>

namespace rm_auto_box
{
    const int RED = 0;
    const int BLUE = 1;
    enum BoxPointType
    {
        POINT_0 = 0,
        POINT_1,
        POINT_2,
        POINT_3,
        CENTER_POINT
    };
    struct Box
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
        bool box_type;
         /*
        @param [0]POINT_0
        @param [1]POINT_1
        @param [2]POINT_2
        @param [3]POINT_3
        @param [4]center_point
        */
       std::vector<cv::Point2f> kpt;
    };
}

#endif // BOX_HPP