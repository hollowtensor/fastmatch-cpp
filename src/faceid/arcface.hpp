#pragma once
#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class ArcFace {
public:
    ArcFace(const std::string& model_path);

    // Extract face embedding given full image + 5-point landmarks
    std::vector<float> get_embedding(const cv::Mat& image,
                                     const std::array<cv::Point2f, 5>& landmarks,
                                     bool normalize = true);

    int embedding_size() const { return embed_dim_; }

private:
    // Align face using landmarks → 112x112 canonical pose
    cv::Mat align_face(const cv::Mat& image,
                       const std::array<cv::Point2f, 5>& landmarks);

    void preprocess(const cv::Mat& aligned, std::vector<float>& blob);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::vector<const char*> input_names_;
    std::string output_name_;
    std::vector<const char*> output_names_;

    int embed_dim_ = 512;

    static constexpr int FACE_SIZE = 112;
    static constexpr float MEAN = 127.5f;
    static constexpr float SCALE = 127.5f;

    // ArcFace reference landmarks for alignment (112x112)
    static constexpr float ref_landmarks_[5][2] = {
        {38.2946f, 51.6963f},
        {73.5318f, 51.5014f},
        {56.0252f, 71.7366f},
        {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };
};
