#include "arcface.hpp"
#include <cmath>

ArcFace::ArcFace(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ArcFace"),
      session_(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), opts);

    auto in_info = session_.GetInputNameAllocated(0, allocator_);
    input_name_ = in_info.get();
    input_names_ = {input_name_.c_str()};

    auto out_info = session_.GetOutputNameAllocated(0, allocator_);
    output_name_ = out_info.get();
    output_names_ = {output_name_.c_str()};

    auto out_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    embed_dim_ = static_cast<int>(out_shape.back());
}

cv::Mat ArcFace::align_face(const cv::Mat& image,
                             const std::array<cv::Point2f, 5>& landmarks) {
    // Source points (detected landmarks)
    cv::Point2f src[5];
    for (int i = 0; i < 5; i++) src[i] = landmarks[i];

    // Destination points (reference alignment for 112x112)
    cv::Point2f dst[5];
    for (int i = 0; i < 5; i++)
        dst[i] = cv::Point2f(ref_landmarks_[i][0], ref_landmarks_[i][1]);

    // Estimate similarity transform using all 5 points
    // Use estimateAffinePartial2D (rotation + scale + translation)
    std::vector<cv::Point2f> src_vec(src, src + 5);
    std::vector<cv::Point2f> dst_vec(dst, dst + 5);
    cv::Mat M = cv::estimateAffinePartial2D(src_vec, dst_vec);

    if (M.empty()) {
        // Fallback: just use first 3 points for affine
        cv::Mat M3 = cv::getAffineTransform(src, dst);
        cv::Mat aligned;
        cv::warpAffine(image, aligned, M3, cv::Size(FACE_SIZE, FACE_SIZE));
        return aligned;
    }

    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, cv::Size(FACE_SIZE, FACE_SIZE));
    return aligned;
}

void ArcFace::preprocess(const cv::Mat& aligned, std::vector<float>& blob) {
    cv::Mat rgb;
    cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);

    blob.resize(3 * FACE_SIZE * FACE_SIZE);
    const int hw = FACE_SIZE * FACE_SIZE;

    for (int y = 0; y < FACE_SIZE; y++) {
        const auto* row = rgb.ptr<cv::Vec3f>(y);
        for (int x = 0; x < FACE_SIZE; x++) {
            blob[0 * hw + y * FACE_SIZE + x] = (row[x][0] - MEAN) / SCALE;
            blob[1 * hw + y * FACE_SIZE + x] = (row[x][1] - MEAN) / SCALE;
            blob[2 * hw + y * FACE_SIZE + x] = (row[x][2] - MEAN) / SCALE;
        }
    }
}

std::vector<float> ArcFace::get_embedding(const cv::Mat& image,
                                           const std::array<cv::Point2f, 5>& landmarks,
                                           bool normalize) {
    cv::Mat aligned = align_face(image, landmarks);

    std::vector<float> blob;
    preprocess(aligned, blob);

    std::array<int64_t, 4> shape = {1, 3, FACE_SIZE, FACE_SIZE};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(), shape.data(), shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_.data(), &tensor, 1,
                                output_names_.data(), output_names_.size());

    const float* data = outputs[0].GetTensorData<float>();
    std::vector<float> embedding(data, data + embed_dim_);

    if (normalize) {
        float norm = 0;
        for (float v : embedding) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-9f)
            for (float& v : embedding) v /= norm;
    }

    return embedding;
}
