#include "feature_extraction.hpp"

FeatureExtraction::FeatureExtraction(const std::string& onnx_path, const std::string& device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "FeatureExtraction"),
      session_(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (device == "cuda") {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

    session_ = Ort::Session(env_, onnx_path.c_str(), opts);

    // Input info
    auto in_info = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = in_info.get();
    input_names_ = {input_name_.c_str()};

    auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    model_h_ = static_cast<int>(shape[2]);
    model_w_ = static_cast<int>(shape[3]);

    // Output info
    auto out_info = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = out_info.get();
    output_names_ = {output_name_.c_str()};

    auto out_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    feat_dim_ = static_cast<int>(out_shape.back());
}

std::vector<float> FeatureExtraction::extract(const cv::Mat& crop) {
    std::vector<float> blob;
    preprocess(crop, blob);

    std::array<int64_t, 4> input_shape = {1, 3, model_h_, model_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.data(), blob.size(), input_shape.data(), input_shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(), &input_tensor, 1,
                                output_names_.data(), output_names_.size());

    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + feat_dim_);
}

void FeatureExtraction::preprocess(const cv::Mat& crop, std::vector<float>& blob) {
    cv::Mat resized, rgb;
    cv::resize(crop, resized, cv::Size(model_w_, model_h_), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    blob.resize(3 * model_h_ * model_w_);
    const int hw = model_h_ * model_w_;

    for (int y = 0; y < model_h_; y++) {
        const auto* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < model_w_; x++) {
            blob[0 * hw + y * model_w_ + x] = (row[x][0] - mean_[0]) / std_[0];
            blob[1 * hw + y * model_w_ + x] = (row[x][1] - mean_[1]) / std_[1];
            blob[2 * hw + y * model_w_ + x] = (row[x][2] - mean_[2]) / std_[2];
        }
    }
}
