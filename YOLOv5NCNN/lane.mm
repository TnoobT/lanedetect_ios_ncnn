#include "lane.h"

bool LaneDetect::hasGPU = false;
bool LaneDetect::toUseGPU = false;
LaneDetect *LaneDetect::detector = nullptr;

LaneDetect::LaneDetect(bool useGPU)
{

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    hasGPU = ncnn::get_gpu_count() > 0;
#endif
    toUseGPU = hasGPU && useGPU;
    
    m_net = new ncnn::Net();
    m_net->opt.use_vulkan_compute = toUseGPU;
    m_net->opt.use_fp16_arithmetic = true;
    NSString *parmaPath = [[NSBundle mainBundle] pathForResource:@"mlsd_no_max_sigmoid_sim" ofType:@"param"];
    NSString *binPath = [[NSBundle mainBundle] pathForResource:@"mlsd_no_max_sigmoid_sim" ofType:@"bin"];
    int rp = m_net->load_param([parmaPath UTF8String]);
    int rm = m_net->load_model([binPath UTF8String]);
    if (rp == 0 && rm == 0) {
        printf("net load param and model success!");
    } else {
        fprintf(stderr, "net load fail,param:%d model:%d", rp, rm);
    }

}


LaneDetect::~LaneDetect()
{   
    m_net->clear();
    delete m_net;
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance();
#endif
}

inline int LaneDetect::clip(float value)
{
    if (value > 0 && value < m_input_size)
        return int(value);
    else if (value < 0)
        return 1;
    else
        return m_input_size - 1;

}



std::vector<LaneDetect::Lanes> LaneDetect::decodeHeatmap(const float* hm,int w, int h)
{   
   // 线段中心点(256*256),线段偏移(4*256*256)
    const float*  displacement = hm+m_hm_size*m_hm_size;
    // exp(center,center);
    std::vector<float> center;
    for (int i = 0;i < m_hm_size*m_hm_size; i++)
    {
        center.push_back( 1/(exp(-hm[i]) + 1) ); // mlsd.mnn原始需要1/(exp(-hm[i]) + 1)
    }
    center.resize(m_hm_size*m_hm_size);

    std::vector<int> index(center.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (center[a] > center[b]); // 从大到小排序
        }
    );
    std::vector<Lanes> lanes;
    
    for (int i = 0; i < index.size(); i++)
    {
        int yy = index[i]/m_hm_size; // 除以宽得行号
        int xx = index[i]%m_hm_size; // 取余宽得列号
        Lanes Lane;
        Lane.x1 = xx + displacement[index[i] + 0*m_hm_size*m_hm_size];
        Lane.y1 = yy + displacement[index[i] + 1*m_hm_size*m_hm_size];
        Lane.x2 = xx + displacement[index[i] + 2*m_hm_size*m_hm_size];
        Lane.y2 = yy + displacement[index[i] + 3*m_hm_size*m_hm_size];
        Lane.lens = sqrt(pow(Lane.x1 - Lane.x2,2) + pow(Lane.y1 - Lane.y2,2));
        Lane.conf = center[index[i]];

        if (center[index[i]] > m_score_thresh && lanes.size() < m_top_k)
        {
            if ( Lane.lens > m_min_len)
            {
                Lane.x1 = clip(w * Lane.x1 / (m_input_size / 2));
                Lane.x2 = clip(w * Lane.x2 / (m_input_size / 2));
                Lane.y1 = clip(h * Lane.y1 / (m_input_size / 2));
                Lane.y2 = clip(h * Lane.y2 / (m_input_size / 2));
                lanes.push_back(Lane);
            }
        }
        else
            break;
    }
    
    return lanes;

}

void LaneDetect::processImg(const cv::Mat& image,ncnn::Mat& in)
{
    int img_w = image.cols;
    int img_h = image.rows;
    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, img_w, img_h);
    in.substract_mean_normalize(m_mean_vals, m_norm_vals);
}

std::vector<LaneDetect::Lanes> LaneDetect::detect(UIImage *image, float score_threshold, float nms_threshold)
{

    
    int width = image.size.width;
    int height = image.size.height;
    auto imageSource = utility::UIImageGetData(image);
    
    
    cv::Mat preImage(m_input_size, m_input_size, CV_8UC3);
    cv::Mat image_input(height, width, CV_8UC4, imageSource.get());
    cv::cvtColor(image_input, image_input, cv::COLOR_RGBA2BGR); // 四通道需要转换！
    cv::resize(image_input, preImage, cv::Size(m_input_size, m_input_size));
    
    
    ncnn::Mat input;
    processImg(preImage, input); // 图片预处理
    auto ex = m_net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    if (toUseGPU) {
        ex.set_vulkan_compute(toUseGPU);
    }
#endif
    ex.input("input", input);
    ncnn::Mat out;
    ex.extract("output", out); //输出,int w, int h
    std::vector<Lanes> lanes = decodeHeatmap(out,width,height);
    
    return lanes;
}
