#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

// Tile camera images into a grid display
cv::Mat stack_images(float scale, const std::vector<std::vector<cv::Mat>>& image_rows);

// Scale bounding box coordinates from one resolution to another
cv::Point scale_coords(cv::Size from, cv::Size to, cv::Point pt);

// Cosine distance between two feature vectors (1.0 = opposite, 0.0 = identical)
float cosine_distance(const std::vector<float>& a, const std::vector<float>& b);

// Which side of directed line (p1->p2) is point (px,py)?
// Returns positive for one side, negative for the other
float line_side(cv::Point p1, cv::Point p2, int px, int py);
