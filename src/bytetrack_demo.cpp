#include <iostream>
#include <chrono>
#include <filesystem>
#include <deque>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "object_detection.hpp"
#include "feature_extraction.hpp"
#include "helpers.hpp"
#include "bytetrack/BYTETracker.h"

// Convert our Detection (model-space coords) to byte_track::Object (display-space)
static byte_track::Object det_to_object(const Detection& det,
                                        cv::Size from, cv::Size to) {
    float sx = static_cast<float>(to.width) / from.width;
    float sy = static_cast<float>(to.height) / from.height;
    float x = det.bbox.x * sx;
    float y = det.bbox.y * sy;
    float w = det.bbox.width * sx;
    float h = det.bbox.height * sy;
    return byte_track::Object(byte_track::Rect<float>(x, y, w, h), 0, det.confidence);
}

// Random but deterministic color per track ID
static cv::Scalar id_color(size_t id) {
    int r = (id * 41 + 97) % 256;
    int g = (id * 73 + 31) % 256;
    int b = (id * 127 + 53) % 256;
    return cv::Scalar(b, g, r);
}

// Re-ID state for a lost track: embedding + original display ID
struct LostTrackEntry {
    size_t display_id;              // the ID shown on screen
    std::vector<float> embedding;   // mean embedding at time of loss
    int lost_frame;                 // frame when lost
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: bytetrack_demo <video_or_webcam_index> [options]\n"
                  << "\nOptions:\n"
                  << "  --config PATH    Config YAML (default: ../config.yaml)\n"
                  << "  --reid           Enable Re-ID for recovering lost tracks\n"
                  << "  --reid-thresh F  Re-ID cosine distance threshold (default: 0.40)\n"
                  << "  --lost-ttl N     Seconds to keep lost tracks for Re-ID (default: 30)\n"
                  << "\nExamples:\n"
                  << "  bytetrack_demo 0                          # webcam, no Re-ID\n"
                  << "  bytetrack_demo 0 --reid                   # webcam + Re-ID\n"
                  << "  bytetrack_demo traffic.mp4 --reid --reid-thresh 0.35\n";
        return 0;
    }

    // Parse args
    std::string source = argv[1];
    std::string config_path = "../config.yaml";
    bool use_reid = false;
    float reid_thresh = 0.40f;
    int lost_ttl_seconds = 30;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--reid") {
            use_reid = true;
        } else if (arg == "--reid-thresh" && i + 1 < argc) {
            reid_thresh = std::stof(argv[++i]);
        } else if (arg == "--lost-ttl" && i + 1 < argc) {
            lost_ttl_seconds = std::stoi(argv[++i]);
        }
    }

    // Load config
    YAML::Node cfg = YAML::LoadFile(config_path);
    auto base = std::filesystem::path(config_path).parent_path();
    auto resolve = [&](const std::string& p) -> std::string {
        std::filesystem::path fp(p);
        return fp.is_absolute() ? p : (base / fp).string();
    };

    std::string det_model = resolve(cfg["object_detection_model_path"].as<std::string>());
    std::string det_classes = resolve(cfg["object_detection_classes_path"].as<std::string>());
    std::string device = cfg["inference_model_device"].as<std::string>("cpu");
    float det_thresh = cfg["object_detection_threshold"].as<float>(0.3f);

    std::cout << "Loading detector: " << det_model << std::endl;
    ObjectDetection detector(det_model, det_classes, device, det_thresh);

    // Re-ID model (only if enabled)
    FeatureExtraction* extractor = nullptr;
    if (use_reid) {
        std::string reid_model = resolve(cfg["feature_extraction_model_path"].as<std::string>());
        reid_thresh = cfg["feature_extraction_threshold"].as<float>(reid_thresh);
        std::cout << "Loading Re-ID model: " << reid_model << std::endl;
        extractor = new FeatureExtraction(reid_model, device);
        std::cout << "Re-ID enabled (thresh=" << reid_thresh << ", lost_ttl=" << lost_ttl_seconds << "s)" << std::endl;
    } else {
        std::cout << "Re-ID disabled (use --reid to enable)" << std::endl;
    }

    // Open source
    cv::VideoCapture cap;
    if (source.size() == 1 && std::isdigit(source[0]))
        cap.open(std::stoi(source));
    else
        cap.open(source);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open: " << source << std::endl;
        return 1;
    }

    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    float fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;
    std::cout << "Source: " << frame_w << "x" << frame_h << " @ " << fps << " fps" << std::endl;

    int lost_ttl_frames = static_cast<int>(fps * lost_ttl_seconds);

    // ByteTrack
    byte_track::BYTETracker tracker(
        static_cast<int>(fps),
        30,                     // track_buffer
        det_thresh,             // track_thresh
        det_thresh + 0.1f,      // high_thresh
        0.8f                    // match_thresh
    );

    cv::Size display_size(frame_w, frame_h);
    cv::Size det_size(detector.model_width(), detector.model_height());

    // Re-ID state
    // bytetrack_id -> display_id mapping (ByteTrack IDs are internal, display IDs persist across re-entries)
    std::unordered_map<size_t, size_t> bt_to_display;
    // bytetrack_id -> current embedding (updated each frame the track is active)
    std::unordered_map<size_t, std::vector<float>> active_embeddings;
    // Lost tracks waiting for re-identification
    std::deque<LostTrackEntry> lost_gallery;
    // Track which bytetrack IDs were active last frame (to detect newly lost tracks)
    std::unordered_map<size_t, bool> prev_active;
    size_t next_display_id = 1;
    int reid_recoveries = 0;

    int frame_num = 0;
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        frame_num++;

        auto t0 = std::chrono::steady_clock::now();

        // Detect
        auto detections = detector.detect(frame);

        // Convert to ByteTrack objects (in display-space coords)
        std::vector<byte_track::Object> objects;
        for (const auto& det : detections)
            objects.push_back(det_to_object(det, det_size, display_size));

        // Track
        auto tracks = tracker.update(objects);

        // Build set of currently active bytetrack IDs
        std::unordered_map<size_t, bool> curr_active;
        for (const auto& track : tracks)
            curr_active[track->getTrackId()] = true;

        if (use_reid) {
            // Detect newly lost tracks: were active last frame, not active now
            for (auto& [bt_id, _] : prev_active) {
                if (!curr_active.count(bt_id) && active_embeddings.count(bt_id)) {
                    size_t disp_id = bt_to_display.count(bt_id) ? bt_to_display[bt_id] : bt_id;
                    lost_gallery.push_back({disp_id, active_embeddings[bt_id], frame_num});
                    active_embeddings.erase(bt_id);
                    std::cout << "  [Re-ID] Track " << disp_id << " lost, saved to gallery" << std::endl;
                }
            }

            // Prune expired entries from lost gallery
            while (!lost_gallery.empty() &&
                   (frame_num - lost_gallery.front().lost_frame) > lost_ttl_frames) {
                lost_gallery.pop_front();
            }

            // For each active track: extract embedding, try Re-ID if new
            for (const auto& track : tracks) {
                size_t bt_id = track->getTrackId();
                const auto& r = track->getRect();
                int x1 = std::max(0, static_cast<int>(r.tl_x()));
                int y1 = std::max(0, static_cast<int>(r.tl_y()));
                int x2 = std::min(frame.cols, static_cast<int>(r.br_x()));
                int y2 = std::min(frame.rows, static_cast<int>(r.br_y()));

                if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;
                cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                auto embedding = extractor->extract(crop);

                // Is this a brand new bytetrack ID we haven't seen?
                bool is_new = !bt_to_display.count(bt_id);

                if (is_new && !lost_gallery.empty()) {
                    // Try to match against lost gallery
                    float best_dist = 1e9f;
                    int best_idx = -1;
                    for (int i = 0; i < static_cast<int>(lost_gallery.size()); i++) {
                        float d = cosine_distance(embedding, lost_gallery[i].embedding);
                        if (d < best_dist) {
                            best_dist = d;
                            best_idx = i;
                        }
                    }

                    if (best_dist < reid_thresh && best_idx >= 0) {
                        // Recovered! Map this bytetrack ID to the old display ID
                        size_t recovered_id = lost_gallery[best_idx].display_id;
                        bt_to_display[bt_id] = recovered_id;
                        lost_gallery.erase(lost_gallery.begin() + best_idx);
                        reid_recoveries++;
                        std::cout << "  [Re-ID] RECOVERED id=" << recovered_id
                                  << " (dist=" << best_dist << ") bt_id=" << bt_id << std::endl;
                    } else {
                        // Genuinely new person
                        bt_to_display[bt_id] = next_display_id++;
                    }
                } else if (is_new) {
                    bt_to_display[bt_id] = next_display_id++;
                }

                // Update active embedding (running average would be better, but keep it simple)
                active_embeddings[bt_id] = embedding;
            }
        }

        prev_active = curr_active;

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Draw tracks
        for (const auto& track : tracks) {
            const auto& r = track->getRect();
            int x1 = static_cast<int>(r.tl_x());
            int y1 = static_cast<int>(r.tl_y());
            int x2 = static_cast<int>(r.br_x());
            int y2 = static_cast<int>(r.br_y());
            size_t bt_id = track->getTrackId();

            // Use display ID if Re-ID is on, otherwise raw bytetrack ID
            size_t disp_id = bt_to_display.count(bt_id) ? bt_to_display[bt_id] : bt_id;
            auto color = id_color(disp_id);

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            char label[48];
            snprintf(label, sizeof(label), "ID %zu (%.0f%%)", disp_id, track->getScore() * 100);
            cv::putText(frame, label, cv::Point(x1, std::max(y1 - 8, 0)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

            cv::circle(frame, cv::Point((x1 + x2) / 2, (y1 + y2) / 2), 4, color, -1);
        }

        // HUD
        char hud[256];
        if (use_reid) {
            snprintf(hud, sizeof(hud), "Frame %d | Dets: %zu | Tracks: %zu | Lost: %zu | Recovered: %d | %.1f ms",
                     frame_num, detections.size(), tracks.size(), lost_gallery.size(), reid_recoveries, ms);
        } else {
            snprintf(hud, sizeof(hud), "Frame %d | Dets: %zu | Tracks: %zu | %.1f ms",
                     frame_num, detections.size(), tracks.size(), ms);
        }
        cv::putText(frame, hud, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);

        if (use_reid) {
            cv::putText(frame, "[Re-ID ON]", cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow("ByteTrack Demo", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete extractor;
    std::cout << "\nDone. " << frame_num << " frames processed.";
    if (use_reid)
        std::cout << " Re-ID recoveries: " << reid_recoveries;
    std::cout << std::endl;
    return 0;
}
