#include "helpers.hpp"
#include <cmath>
#include <numeric>

cv::Mat stack_images(float scale, const std::vector<std::vector<cv::Mat>>& image_rows) {
    if (image_rows.empty() || image_rows[0].empty())
        return cv::Mat();

    std::vector<cv::Mat> row_images;
    for (const auto& row : image_rows) {
        std::vector<cv::Mat> scaled;
        for (const auto& img : row) {
            cv::Mat s;
            cv::resize(img, s, cv::Size(), scale, scale);
            if (s.channels() == 1)
                cv::cvtColor(s, s, cv::COLOR_GRAY2BGR);
            scaled.push_back(s);
        }
        cv::Mat hcat;
        cv::hconcat(scaled, hcat);
        row_images.push_back(hcat);
    }

    // Ensure all rows have same width
    int max_w = 0;
    for (const auto& r : row_images)
        max_w = std::max(max_w, r.cols);

    for (auto& r : row_images) {
        if (r.cols < max_w) {
            cv::Mat padded = cv::Mat::zeros(r.rows, max_w, r.type());
            r.copyTo(padded(cv::Rect(0, 0, r.cols, r.rows)));
            r = padded;
        }
    }

    cv::Mat result;
    cv::vconcat(row_images, result);
    return result;
}

cv::Point scale_coords(cv::Size from, cv::Size to, cv::Point pt) {
    int x = static_cast<int>(pt.x * static_cast<float>(to.width) / from.width);
    int y = static_cast<int>(pt.y * static_cast<float>(to.height) / from.height);
    return cv::Point(std::max(0, x), std::max(0, y));
}

float cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-9f) return 1.0f;
    return 1.0f - dot / denom;
}

float line_side(cv::Point p1, cv::Point p2, int px, int py) {
    return static_cast<float>((p2.x - p1.x) * (py - p1.y) - (p2.y - p1.y) * (px - p1.x));
}
