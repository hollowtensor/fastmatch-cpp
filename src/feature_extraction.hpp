#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class FeatureExtraction {
public:
    FeatureExtraction(const std::string& onnx_path, const std::string& device);

    // Returns a 512-d (or model-dependent) feature vector
    std::vector<float> extract(const cv::Mat& crop);

    int feature_dim() const { return feat_dim_; }

private:
    void preprocess(const cv::Mat& crop, std::vector<float>& blob);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::vector<const char*> input_names_;
    std::string output_name_;
    std::vector<const char*> output_names_;

    int model_h_ = 256;
    int model_w_ = 128;
    int feat_dim_ = 512;

    // ImageNet normalization constants
    static constexpr float mean_[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float std_[3] = {0.229f, 0.224f, 0.225f};
};
