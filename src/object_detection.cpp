#include "object_detection.hpp"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

ObjectDetection::ObjectDetection(const std::string& onnx_path,
                                 const std::string& coco_names_path,
                                 const std::string& device,
                                 float confidence_threshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ObjectDetection"),
      session_(nullptr),
      conf_threshold_(confidence_threshold),
      nms_threshold_(std::max(0.0f, confidence_threshold - 0.1f))
{
    // Load class names
    std::ifstream f(coco_names_path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        class_names_.push_back(line);
    }

    // Session options
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

    session_ = Ort::Session(env_, onnx_path.c_str(), opts);

    // Get input/output names and dimensions
    auto input_info = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = input_info.get();
    input_names_ = {input_name_.c_str()};

    auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    model_h_ = static_cast<int>(shape[2]);
    model_w_ = static_cast<int>(shape[3]);

    size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        if (i == 0) output_name0_ = name.get();
        else output_name1_ = name.get();
    }
    output_names_ = {output_name0_.c_str(), output_name1_.c_str()};
}

std::vector<Detection> ObjectDetection::detect(const cv::Mat& frame) {
    std::vector<float> blob;
    preprocess(frame, blob);

    std::array<int64_t, 4> input_shape = {1, 3, model_h_, model_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(), &input_tensor, 1,
                                output_names_.data(), output_names_.size());

    return postprocess(outputs);
}

void ObjectDetection::preprocess(const cv::Mat& frame, std::vector<float>& blob) {
    cv::Mat resized, rgb;
    cv::resize(frame, resized, cv::Size(model_w_, model_h_), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    blob.resize(3 * model_h_ * model_w_);
    const int hw = model_h_ * model_w_;

    for (int y = 0; y < model_h_; y++) {
        const auto* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < model_w_; x++) {
            blob[0 * hw + y * model_w_ + x] = row[x][0] / 255.0f;
            blob[1 * hw + y * model_w_ + x] = row[x][1] / 255.0f;
            blob[2 * hw + y * model_w_ + x] = row[x][2] / 255.0f;
        }
    }
}

std::vector<Detection> ObjectDetection::postprocess(const std::vector<Ort::Value>& outputs) {
    // output[0] = boxes [1, N, 1, 4], output[1] = confs [1, N, num_classes]
    auto box_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto conf_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    const float* box_data = outputs[0].GetTensorData<float>();
    const float* conf_data = outputs[1].GetTensorData<float>();

    int num_boxes = static_cast<int>(box_shape[1]);
    int num_classes = static_cast<int>(conf_shape[2]);

    // Per-class NMS
    std::vector<Detection> results;

    // Group by class
    for (int cls = 0; cls < num_classes; cls++) {
        // Check if this is a class we care about ("person")
        if (cls >= (int)class_names_.size()) continue;
        std::string name = class_names_[cls];
        // Capitalize first letter to match Python behavior
        if (!name.empty()) name[0] = std::toupper(name[0]);
        if (name != "Person") continue;

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;

        for (int i = 0; i < num_boxes; i++) {
            float conf = conf_data[i * num_classes + cls];
            if (conf <= conf_threshold_) continue;

            // Check this class is the max for this box
            float max_conf = 0;
            for (int c = 0; c < num_classes; c++)
                max_conf = std::max(max_conf, conf_data[i * num_classes + c]);
            if (conf < max_conf) continue;

            float x1 = box_data[i * 4 + 0] * model_w_;
            float y1 = box_data[i * 4 + 1] * model_h_;
            float x2 = box_data[i * 4 + 2] * model_w_;
            float y2 = box_data[i * 4 + 3] * model_h_;

            boxes.emplace_back(cv::Rect(
                static_cast<int>(x1), static_cast<int>(y1),
                static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)));
            scores.push_back(conf);
        }

        auto keep = nms(boxes, scores, nms_threshold_);
        for (int idx : keep) {
            Detection det;
            det.bbox = cv::Rect(
                boxes[idx].x, boxes[idx].y,
                boxes[idx].width, boxes[idx].height);
            det.confidence = scores[idx];
            det.class_name = name;
            results.push_back(det);
        }
    }

    return results;
}

std::vector<int> ObjectDetection::nms(const std::vector<cv::Rect>& boxes,
                                       const std::vector<float>& scores,
                                       float threshold) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(scores.size(), false);

    for (int i : indices) {
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (int j : indices) {
            if (suppressed[j] || j == i) continue;
            int xx1 = std::max(boxes[i].x, boxes[j].x);
            int yy1 = std::max(boxes[i].y, boxes[j].y);
            int xx2 = std::min(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
            int yy2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);
            float inter = std::max(0, xx2 - xx1) * std::max(0, yy2 - yy1);
            float area_i = boxes[i].width * boxes[i].height;
            float area_j = boxes[j].width * boxes[j].height;
            float iou = inter / (area_i + area_j - inter);
            if (iou > threshold) suppressed[j] = true;
        }
    }
    return keep;
}
