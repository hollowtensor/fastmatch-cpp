#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect bbox;
    float confidence;
    std::string class_name;
};

class ObjectDetection {
public:
    ObjectDetection(const std::string& onnx_path,
                    const std::string& coco_names_path,
                    const std::string& device,
                    float confidence_threshold = 0.3f);

    std::vector<Detection> detect(const cv::Mat& frame);

    int model_width() const { return model_w_; }
    int model_height() const { return model_h_; }

private:
    void preprocess(const cv::Mat& frame, std::vector<float>& blob);
    std::vector<Detection> postprocess(const std::vector<Ort::Value>& outputs);
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float threshold);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> class_names_;
    std::string input_name_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::string output_name0_, output_name1_;

    int model_w_ = 416;
    int model_h_ = 416;
    float conf_threshold_;
    float nms_threshold_;
};
