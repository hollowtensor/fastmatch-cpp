#include "scrfd.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

SCRFD::SCRFD(const std::string& model_path, cv::Size input_size,
             float conf_threshold, float nms_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "SCRFD"),
      session_(nullptr),
      input_size_(input_size),
      conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), opts);

    // Input
    auto in_info = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = in_info.get();
    input_names_ = {input_name_.c_str()};

    // Outputs (9 total: 3 strides x [scores, boxes, keypoints])
    size_t num_out = session_.GetOutputCount();
    for (size_t i = 0; i < num_out; i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_name_strs_.push_back(name.get());
    }
    for (auto& s : output_name_strs_)
        output_names_.push_back(s.c_str());
}

void SCRFD::forward(const cv::Mat& image, float det_scale,
                    std::vector<float>& scores_out,
                    std::vector<cv::Rect2f>& boxes_out,
                    std::vector<std::array<cv::Point2f, 5>>& kps_out) {
    // Preprocess: blobFromImage with mean/std
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / STD, image.size(),
                           cv::Scalar(MEAN, MEAN, MEAN), true);

    int input_h = image.rows;
    int input_w = image.cols;

    // Create tensor
    std::array<int64_t, 4> shape = {1, 3, input_h, input_w};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        mem, (float*)blob.data, blob.total(), shape.data(), shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(), &tensor, 1,
                                output_names_.data(), output_names_.size());

    // Parse outputs: idx 0..2 = scores, 3..5 = boxes, 6..8 = keypoints
    for (int idx = 0; idx < FMC; idx++) {
        int stride = strides_[idx];

        const float* score_data = outputs[idx].GetTensorData<float>();
        const float* bbox_data = outputs[idx + FMC].GetTensorData<float>();
        const float* kps_data = outputs[idx + FMC * 2].GetTensorData<float>();

        auto score_shape = outputs[idx].GetTensorTypeAndShapeInfo().GetShape();
        int num_anchors_total = static_cast<int>(score_shape[0]);

        int grid_h = input_h / stride;
        int grid_w = input_w / stride;

        // Get or build anchor centers
        GridKey key{grid_h, grid_w, stride};
        if (anchor_cache_.find(key) == anchor_cache_.end()) {
            std::vector<cv::Point2f> centers;
            centers.reserve(grid_h * grid_w * NUM_ANCHORS);
            for (int y = 0; y < grid_h; y++) {
                for (int x = 0; x < grid_w; x++) {
                    for (int a = 0; a < NUM_ANCHORS; a++) {
                        centers.emplace_back(x * stride, y * stride);
                    }
                }
            }
            anchor_cache_[key] = std::move(centers);
        }
        const auto& anchors = anchor_cache_[key];

        for (int i = 0; i < num_anchors_total; i++) {
            float score = score_data[i];
            if (score < conf_threshold_) continue;

            float cx = anchors[i].x;
            float cy = anchors[i].y;

            // Decode bbox: distance to left, top, right, bottom
            float x1 = (cx - bbox_data[i * 4 + 0] * stride) / det_scale;
            float y1 = (cy - bbox_data[i * 4 + 1] * stride) / det_scale;
            float x2 = (cx + bbox_data[i * 4 + 2] * stride) / det_scale;
            float y2 = (cy + bbox_data[i * 4 + 3] * stride) / det_scale;

            scores_out.push_back(score);
            boxes_out.emplace_back(x1, y1, x2 - x1, y2 - y1);

            // Decode 5 keypoints
            std::array<cv::Point2f, 5> kps;
            for (int k = 0; k < 5; k++) {
                kps[k].x = (cx + kps_data[i * 10 + k * 2] * stride) / det_scale;
                kps[k].y = (cy + kps_data[i * 10 + k * 2 + 1] * stride) / det_scale;
            }
            kps_out.push_back(kps);
        }
    }
}

std::vector<FaceDetection> SCRFD::detect(const cv::Mat& image, int max_num) {
    // Letterbox resize maintaining aspect ratio
    float im_ratio = static_cast<float>(image.rows) / image.cols;
    float model_ratio = static_cast<float>(input_size_.height) / input_size_.width;

    int new_w, new_h;
    if (im_ratio > model_ratio) {
        new_h = input_size_.height;
        new_w = static_cast<int>(new_h / im_ratio);
    } else {
        new_w = input_size_.width;
        new_h = static_cast<int>(new_w * im_ratio);
    }

    float det_scale = static_cast<float>(new_h) / image.rows;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    cv::Mat det_image = cv::Mat::zeros(input_size_.height, input_size_.width, CV_8UC3);
    resized.copyTo(det_image(cv::Rect(0, 0, new_w, new_h)));

    // Forward pass
    std::vector<float> scores;
    std::vector<cv::Rect2f> boxes;
    std::vector<std::array<cv::Point2f, 5>> kps;
    forward(det_image, det_scale, scores, boxes, kps);

    // Sort by score descending
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    // NMS
    auto keep = nms(boxes, scores, nms_threshold_);

    // Build results
    std::vector<FaceDetection> results;
    for (int idx : keep) {
        FaceDetection det;
        det.bbox = boxes[idx];
        det.score = scores[idx];
        det.landmarks = kps[idx];
        results.push_back(det);
    }

    // Sort by area (largest first) and limit
    if (max_num > 0 && static_cast<int>(results.size()) > max_num) {
        std::sort(results.begin(), results.end(), [](const FaceDetection& a, const FaceDetection& b) {
            return (a.bbox.width * a.bbox.height) > (b.bbox.width * b.bbox.height);
        });
        results.resize(max_num);
    }

    return results;
}

std::vector<int> SCRFD::nms(const std::vector<cv::Rect2f>& boxes,
                             const std::vector<float>& scores,
                             float threshold) {
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(scores.size(), false);
    std::vector<int> keep;

    for (int i : order) {
        if (suppressed[i]) continue;
        keep.push_back(i);

        for (int j : order) {
            if (suppressed[j] || j == i) continue;
            float xx1 = std::max(boxes[i].x, boxes[j].x);
            float yy1 = std::max(boxes[i].y, boxes[j].y);
            float xx2 = std::min(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
            float yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);
            float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
            float area_i = boxes[i].width * boxes[i].height;
            float area_j = boxes[j].width * boxes[j].height;
            float iou = inter / (area_i + area_j - inter);
            if (iou > threshold) suppressed[j] = true;
        }
    }
    return keep;
}
