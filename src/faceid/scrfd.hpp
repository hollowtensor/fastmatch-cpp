#pragma once
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct FaceDetection {
    cv::Rect2f bbox;
    float score;
    // 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    std::array<cv::Point2f, 5> landmarks;
};

class SCRFD {
public:
    SCRFD(const std::string& model_path,
          cv::Size input_size = {640, 640},
          float conf_threshold = 0.5f,
          float nms_threshold = 0.4f);

    std::vector<FaceDetection> detect(const cv::Mat& image, int max_num = 0);

private:
    void forward(const cv::Mat& image, float det_scale,
                 std::vector<float>& scores_out,
                 std::vector<cv::Rect2f>& boxes_out,
                 std::vector<std::array<cv::Point2f, 5>>& kps_out);

    std::vector<int> nms(const std::vector<cv::Rect2f>& boxes,
                         const std::vector<float>& scores,
                         float threshold);

    Ort::Env env_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string input_name_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_name_strs_;
    std::vector<const char*> output_names_;

    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;

    static constexpr int FMC = 3;
    static constexpr int NUM_ANCHORS = 2;
    static constexpr int strides_[3] = {8, 16, 32};
    static constexpr float MEAN = 127.5f;
    static constexpr float STD = 128.0f;

    // Cached anchor centers per (height, width, stride)
    struct GridKey {
        int h, w, stride;
        bool operator==(const GridKey& o) const { return h == o.h && w == o.w && stride == o.stride; }
    };
    struct GridHash {
        size_t operator()(const GridKey& k) const {
            return std::hash<int>()(k.h) ^ (std::hash<int>()(k.w) << 16) ^ (std::hash<int>()(k.stride) << 8);
        }
    };
    std::unordered_map<GridKey, std::vector<cv::Point2f>, GridHash> anchor_cache_;
};
