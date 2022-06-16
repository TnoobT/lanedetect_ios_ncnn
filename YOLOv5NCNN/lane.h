#ifndef __LANE_H__
#define __LANE_H__

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <UIKit/UIImage.h>
#import <functional>
#include "ncnn/ncnn/net.h"
#include "UIImage+Utility.h"

class LaneDetect
{
    public:
        typedef struct Lanes
        {
            float x1;  // 起点
            float y1;
            float x2;  // 终点
            float y2;
            float lens;
            float conf;

        }Lanes;
    
    public:
        LaneDetect(bool useGPU);
        ~LaneDetect();   
        std::vector<Lanes> detect(UIImage *image, float score_threshold, float nms_threshold);
        std::vector<Lanes> decodeHeatmap(const float* heatmap,int w, int h);
        void processImg(const cv::Mat& image,ncnn::Mat& in);
        inline int clip(float value);
    
    private:
        ncnn::Net *m_net = nullptr;

        std::vector<Lanes> m_lanes;
        const float m_mean_vals[3] = { 127.5f, 127.5f, 127.5f };
        const float m_norm_vals[3] = { 1/127.5f, 1/127.5f, 1/127.5f };
        float m_score_thresh = 0.2; // 阈值
        int m_input_size     = 512; // 输入尺寸
        int m_hm_size  = 256; // 特征图大小
        int m_min_len  = 20; // 预测线段的最短长度
        int m_top_k    = 200; // 取200条线
    public:
        static LaneDetect *detector;
        static bool hasGPU;
        static bool toUseGPU;

};






#endif
