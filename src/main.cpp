#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "calibration.hpp"
#include "object_detection.hpp"
#include "feature_extraction.hpp"
#include "helpers.hpp"

namespace fs = std::filesystem;

struct PersonRecord {
    int id;
    int camera_id;
    std::string cls_name;
    cv::Rect bbox;
    float confidence;
    cv::Scalar color;
    std::deque<std::vector<float>> gallery;
};

// Open camera sources based on CLI options
std::vector<cv::VideoCapture> open_sources(const RuntimeOpts& opts) {
    std::vector<cv::VideoCapture> cameras;
    cv::Size sz(opts.width, opts.height);

    auto open_cap = [&](cv::VideoCapture cap, const std::string& label) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, sz.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, sz.height);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Cannot open " << label << std::endl;
            return false;
        }
        std::cout << "[cam" << cameras.size() << "] " << label << std::endl;
        cameras.push_back(std::move(cap));
        return true;
    };

    switch (opts.source_type) {
        case RuntimeOpts::WEBCAM:
            if (!open_cap(cv::VideoCapture(opts.webcam_index), "Webcam " + std::to_string(opts.webcam_index)))
                std::exit(1);
            break;

        case RuntimeOpts::RTSP:
            for (const auto& url : opts.rtsp_urls) {
                if (!open_cap(cv::VideoCapture(url), "RTSP: " + url))
                    std::exit(1);
            }
            break;

        case RuntimeOpts::VIDEO_FILE:
            if (!open_cap(cv::VideoCapture(opts.video_path), "Video: " + opts.video_path))
                std::exit(1);
            break;

        case RuntimeOpts::VIDEO_DIR: {
            std::vector<std::string> exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"};
            for (const auto& entry : fs::directory_iterator(opts.video_path)) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
                    open_cap(cv::VideoCapture(entry.path().string()),
                             "Video: " + entry.path().filename().string());
                }
            }
            break;
        }
        default:
            break;
    }

    return cameras;
}

int main(int argc, char** argv) {
    RuntimeOpts opts = RuntimeOpts::parse(argc, argv);
    ModelConfig mcfg = ModelConfig::load(opts.config_path);

    // Init models
    std::cout << "Loading detection model: " << mcfg.detection_model << std::endl;
    ObjectDetection detector(mcfg.detection_model, mcfg.detection_classes,
                             mcfg.device, mcfg.detection_threshold);

    std::cout << "Loading re-id model: " << mcfg.reid_model << std::endl;
    FeatureExtraction extractor(mcfg.reid_model, mcfg.device);

    // Default to webcam if no source specified
    if (opts.source_type == RuntimeOpts::NONE) {
        opts.source_type = RuntimeOpts::WEBCAM;
        std::cout << "No source specified, defaulting to webcam." << std::endl;
    }

    // If --rtsp with no URLs, let user pick from config
    if (opts.source_type == RuntimeOpts::RTSP && opts.rtsp_urls.empty()) {
        if (mcfg.rtsp_streams.empty()) {
            std::cerr << "No RTSP streams in config.yaml" << std::endl;
            return 1;
        }
        if (mcfg.rtsp_streams.size() == 1) {
            opts.rtsp_urls = mcfg.rtsp_streams;
        } else {
            std::cout << "\nSaved RTSP streams:\n";
            for (size_t i = 0; i < mcfg.rtsp_streams.size(); i++)
                std::cout << "  [" << i + 1 << "] " << mcfg.rtsp_streams[i] << "\n";
            std::cout << "  [a] All streams\n";
            std::cout << "\nSelect (1-" << mcfg.rtsp_streams.size() << ", or 'a' for all): ";

            std::string input;
            std::getline(std::cin, input);

            if (input == "a" || input == "A") {
                opts.rtsp_urls = mcfg.rtsp_streams;
            } else {
                int idx = std::stoi(input) - 1;
                if (idx < 0 || idx >= static_cast<int>(mcfg.rtsp_streams.size())) {
                    std::cerr << "Invalid selection." << std::endl;
                    return 1;
                }
                opts.rtsp_urls.push_back(mcfg.rtsp_streams[idx]);
            }
        }
    }

    // Open sources
    auto cameras = open_sources(opts);
    int total_cam = static_cast<int>(cameras.size());
    std::cout << "\nTotal cameras: " << total_cam << std::endl;
    if (total_cam == 0) {
        std::cerr << "No camera sources opened." << std::endl;
        return 1;
    }

    // Interactive calibration or load saved
    if (opts.calibrate) {
        auto cal = calibration::run_calibration(cameras[0], cv::Size(opts.width, opts.height));
        if (cal.has_zone) {
            opts.zone_enabled = true;
            opts.zone_points = cal.zone_points;
        }
        if (cal.has_line) {
            opts.line_enabled = true;
            opts.line_p1 = cal.line_p1;
            opts.line_p2 = cal.line_p2;
            opts.entry_sign = cal.entry_sign;
        }
    } else if (!opts.zone_enabled && !opts.line_enabled) {
        // Auto-load previous calibration if no zone/line specified via CLI
        CalibrationResult saved;
        if (CalibrationResult::load("calibration.yaml", saved)) {
            if (saved.has_zone) {
                opts.zone_enabled = true;
                opts.zone_points = saved.zone_points;
            }
            if (saved.has_line) {
                opts.line_enabled = true;
                opts.line_p1 = saved.line_p1;
                opts.line_p2 = saved.line_p2;
                opts.entry_sign = saved.entry_sign;
            }
        }
    }

    // Zone
    std::vector<cv::Point> zone_polygon;
    if (opts.zone_enabled && !opts.zone_points.empty()) {
        zone_polygon = opts.zone_points;
        std::cout << "Zone: " << zone_polygon.size() << " vertices" << std::endl;
    }

    // Entry/exit line
    cv::Point line_p1 = opts.line_p1, line_p2 = opts.line_p2;
    int entry_sign = opts.entry_sign;
    if (opts.line_enabled)
        std::cout << "Line: (" << line_p1.x << "," << line_p1.y << ")->("
                  << line_p2.x << "," << line_p2.y << ") sign=" << entry_sign << std::endl;

    // Tracking state
    std::unordered_map<int, PersonRecord> persons;
    int next_id = 0, frame_num = 0;
    int entry_count = 0, exit_count = 0;
    std::unordered_map<int, std::string> person_side;
    std::unordered_map<int, int> person_last_crossed;
    std::unordered_map<int, int> person_last_seen;
    std::deque<std::string> person_events;
    const int CROSSING_COOLDOWN = 3;  // low to catch fast crossings

    const cv::Size cam_size(opts.width, opts.height);
    const cv::Size det_size(detector.model_width(), detector.model_height());

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> color_dist(0, 255);

    // Video writer
    cv::VideoWriter writer;
    if (opts.save) {
        cv::Mat dummy = cv::Mat::zeros(opts.height, opts.width, CV_8UC3);
        std::vector<std::vector<cv::Mat>> rows;
        if (total_cam % 2 == 0) {
            rows.resize(2);
            for (int i = 0; i < total_cam / 2; i++) rows[0].push_back(dummy);
            for (int i = total_cam / 2; i < total_cam; i++) rows[1].push_back(dummy);
        } else {
            rows.resize(1);
            for (int i = 0; i < total_cam; i++) rows[0].push_back(dummy);
        }
        cv::Mat test = stack_images(opts.display_scale, rows);
        writer.open(opts.output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    opts.output_fps, test.size());
        std::cout << "Saving to: " << opts.output_path << std::endl;
    }

    // Main loop
    while (true) {
        auto t_start = std::chrono::steady_clock::now();
        frame_num++;

        // Read frames
        std::vector<cv::Mat> frames(total_cam);
        bool done = false;
        for (int i = 0; i < total_cam; i++) {
            if (!cameras[i].read(frames[i])) {
                std::cout << "Camera " << i << " stream ended." << std::endl;
                done = true;
                break;
            }
        }
        if (done) break;

        // Detect + resize
        std::vector<std::vector<Detection>> all_dets(total_cam);
        std::vector<cv::Mat> display_frames(total_cam);

        for (int i = 0; i < total_cam; i++) {
            all_dets[i] = detector.detect(frames[i]);
            cv::resize(frames[i], display_frames[i], cam_size, 0, 0, cv::INTER_CUBIC);
        }

        int total_detections = 0;
        for (auto& d : all_dets) total_detections += static_cast<int>(d.size());

        if (total_detections > 0 || frame_num % 100 == 0) {
            std::cout << "\n[Frame " << frame_num << "] Detections: " << total_detections;
            for (int i = 0; i < total_cam; i++)
                std::cout << " cam" << i << "=" << all_dets[i].size();
            std::cout << std::endl;
        }

        std::unordered_set<int> active_this_frame;

        for (int cam_i = 0; cam_i < total_cam; cam_i++) {
            for (const auto& det : all_dets[cam_i]) {
                cv::Point tl = scale_coords(det_size, cam_size,
                    cv::Point(det.bbox.x, det.bbox.y));
                cv::Point br = scale_coords(det_size, cam_size,
                    cv::Point(det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height));

                // Zone filter
                if (!zone_polygon.empty()) {
                    int cx = (tl.x + br.x) / 2, cy = (tl.y + br.y) / 2;
                    if (cv::pointPolygonTest(zone_polygon, cv::Point2f(cx, cy), false) < 0)
                        continue;
                }

                // Crop + extract
                cv::Rect crop_rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
                crop_rect &= cv::Rect(0, 0, display_frames[cam_i].cols, display_frames[cam_i].rows);
                if (crop_rect.width <= 0 || crop_rect.height <= 0) continue;

                cv::Mat crop = display_frames[cam_i](crop_rect);
                auto features = extractor.extract(crop);

                if (persons.empty()) {
                    PersonRecord rec;
                    rec.id = next_id; rec.camera_id = cam_i;
                    rec.cls_name = det.class_name;
                    rec.bbox = cv::Rect(tl, br);
                    rec.confidence = det.confidence;
                    rec.color = cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng));
                    rec.gallery.push_back(features);
                    persons[next_id] = std::move(rec);
                    std::cout << "  [cam" << cam_i << "] NEW id=" << next_id
                              << " conf=" << det.confidence << std::endl;
                    active_this_frame.insert(next_id);
                    next_id++;
                } else {
                    int best_id = -1;
                    float best_dist = 1e9f;

                    for (auto& [pid, rec] : persons) {
                        int n = static_cast<int>(rec.gallery.size());
                        std::vector<int> indices;
                        if (n > 50) {
                            for (int j = std::max(0, n - 20); j < n; j++) indices.push_back(j);
                            int older = n - 20;
                            if (older > 0) {
                                int step = std::max(1, older / 30);
                                for (int j = 0; j < older; j += step) indices.push_back(j);
                            }
                            std::sort(indices.begin(), indices.end());
                            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
                        } else {
                            indices.resize(n);
                            std::iota(indices.begin(), indices.end(), 0);
                        }

                        float min_d = 1e9f;
                        for (int idx : indices)
                            min_d = std::min(min_d, cosine_distance(rec.gallery[idx], features));
                        if (min_d < best_dist) { best_dist = min_d; best_id = pid; }
                    }

                    if (best_dist < mcfg.reid_threshold) {
                        auto& rec = persons[best_id];
                        std::cout << "  [cam" << cam_i << "] MATCH id=" << best_id
                                  << " dist=" << best_dist << " gallery=" << rec.gallery.size() << std::endl;
                        if (static_cast<int>(rec.gallery.size()) < mcfg.max_gallery)
                            rec.gallery.push_back(features);
                        else { rec.gallery.pop_front(); rec.gallery.push_back(features); }
                        rec.camera_id = cam_i;
                        rec.bbox = cv::Rect(tl, br);
                        rec.confidence = det.confidence;
                        active_this_frame.insert(best_id);
                    } else {
                        PersonRecord rec;
                        rec.id = next_id; rec.camera_id = cam_i;
                        rec.cls_name = det.class_name;
                        rec.bbox = cv::Rect(tl, br);
                        rec.confidence = det.confidence;
                        rec.color = cv::Scalar(color_dist(rng), color_dist(rng), color_dist(rng));
                        rec.gallery.push_back(features);
                        persons[next_id] = std::move(rec);
                        std::cout << "  [cam" << cam_i << "] NEW id=" << next_id
                                  << " best=" << best_dist << std::endl;
                        active_this_frame.insert(next_id);
                        next_id++;
                    }
                }
            }

            // Line crossing (only if line is configured)
            if (opts.line_enabled) {
                for (auto& [pid, rec] : persons) {
                    if (rec.camera_id != cam_i || !active_this_frame.count(pid)) continue;

                    int cx = rec.bbox.x + rec.bbox.width / 2;
                    int cy = rec.bbox.y + rec.bbox.height / 2;
                    float sv = line_side(line_p1, line_p2, cx, cy);
                    std::string current = (sv > 0) ? "A" : "B";

                    if (person_side.count(pid)) {
                        const auto& prev = person_side[pid];
                        int since = frame_num - person_last_crossed.count(pid) ?
                            frame_num - person_last_crossed[pid] : 999;
                        if (prev != current && since > CROSSING_COOLDOWN) {
                            bool is_entry = (entry_sign > 0 && prev == "A") ||
                                            (entry_sign < 0 && prev == "B");
                            if (is_entry) { entry_count++; person_events.push_back("ENTRY id=" + std::to_string(pid)); }
                            else { exit_count++; person_events.push_back("EXIT id=" + std::to_string(pid)); }
                            person_last_crossed[pid] = frame_num;
                            std::cout << "  >> " << person_events.back() << std::endl;
                            if (person_events.size() > 5) person_events.pop_front();
                        }
                    }
                    person_side[pid] = current;
                    person_last_seen[pid] = frame_num;
                }
            }

            // Draw bboxes
            for (auto& [pid, rec] : persons) {
                if (rec.camera_id != cam_i || !active_this_frame.count(pid)) continue;

                cv::rectangle(display_frames[cam_i], rec.bbox, rec.color, 2);
                char lbl[64];
                snprintf(lbl, sizeof(lbl), "%s %d: %.2f", rec.cls_name.c_str(), rec.id, rec.confidence);
                cv::putText(display_frames[cam_i], lbl,
                    cv::Point(rec.bbox.x, std::max(rec.bbox.y - 10, 0)),
                    cv::FONT_HERSHEY_PLAIN, 1, rec.color, 2);

                int acx = rec.bbox.x + rec.bbox.width / 2;
                int acy = rec.bbox.y + rec.bbox.height / 2;

                if (opts.line_enabled) {
                    const auto& ps = person_side[pid];
                    bool is_entry_side = (entry_sign > 0 && ps == "A") || (entry_sign < 0 && ps == "B");
                    bool recently_crossed = person_last_crossed.count(pid) &&
                        (frame_num - person_last_crossed[pid] < 30);
                    cv::Scalar dot_color = recently_crossed ? cv::Scalar(0, 255, 255) :
                        is_entry_side ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 255);
                    cv::circle(display_frames[cam_i], cv::Point(acx, acy), 7, dot_color, -1);
                    cv::circle(display_frames[cam_i], cv::Point(acx, acy), 7, cv::Scalar(255, 255, 255), 1);
                    cv::putText(display_frames[cam_i], is_entry_side ? "ENTRY" : "EXIT",
                        cv::Point(acx + 10, acy + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, dot_color, 1);
                } else {
                    cv::circle(display_frames[cam_i], cv::Point(acx, acy), 5, rec.color, -1);
                }
            }
        }

        // FPS
        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        double fps = 1.0 / (elapsed + 1e-9);

        if (total_detections > 0 || frame_num % 100 == 0)
            std::cout << "  Tracked: " << persons.size() << " | FPS: " << fps << std::endl;

        // Build display
        std::vector<std::vector<cv::Mat>> grid;
        if (total_cam % 2 == 0) {
            grid.resize(2);
            for (int i = 0; i < total_cam / 2; i++) grid[0].push_back(display_frames[i]);
            for (int i = total_cam / 2; i < total_cam; i++) grid[1].push_back(display_frames[i]);
        } else {
            grid.resize(1);
            for (int i = 0; i < total_cam; i++) grid[0].push_back(display_frames[i]);
        }

        cv::Mat display = stack_images(opts.display_scale, grid);
        int h = display.rows, w = display.cols;

        // HUD
        cv::Mat overlay = display.clone();
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(w, 65), cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.6, display, 0.4, 0, display);

        char buf[256];
        snprintf(buf, sizeof(buf), "FPS: %.1f  |  Detections: %d  |  Tracked: %zu",
                 fps, total_detections, persons.size());
        cv::putText(display, buf, cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        if (opts.line_enabled) {
            snprintf(buf, sizeof(buf), "Entries: %d  |  Exits: %d  |  Inside: %d",
                     entry_count, exit_count, entry_count - exit_count);
            cv::putText(display, buf, cv::Point(15, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            // Draw line
            float sc = opts.display_scale;
            cv::Point lp1(int(line_p1.x * sc), int(line_p1.y * sc));
            cv::Point lp2(int(line_p2.x * sc), int(line_p2.y * sc));
            cv::line(display, lp1, lp2, cv::Scalar(0, 255, 255), 2);

            int mx = (lp1.x + lp2.x) / 2, my = (lp1.y + lp2.y) / 2;
            float dx = float(lp2.x - lp1.x), dy = float(lp2.y - lp1.y);
            float len = std::max(std::sqrt(dx * dx + dy * dy), 1.0f);
            float nx = -dy / len * 30, ny = dx / len * 30;
            cv::putText(display, "ENTRY",
                cv::Point(int(mx + nx * entry_sign) - 30, int(my + ny * entry_sign)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            cv::putText(display, "EXIT",
                cv::Point(int(mx - nx * entry_sign) - 25, int(my - ny * entry_sign)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // Zone overlay
        if (!zone_polygon.empty()) {
            float sc = opts.display_scale;
            std::vector<cv::Point> sz;
            for (const auto& pt : zone_polygon) sz.emplace_back(int(pt.x * sc), int(pt.y * sc));
            cv::Mat zo = display.clone();
            cv::fillPoly(zo, std::vector<std::vector<cv::Point>>{sz}, cv::Scalar(0, 180, 0));
            cv::addWeighted(zo, 0.15, display, 0.85, 0, display);
            cv::polylines(display, std::vector<std::vector<cv::Point>>{sz}, true, cv::Scalar(0, 255, 0), 2);
        }

        // Event log
        if (!person_events.empty()) {
            cv::Mat ov2 = display.clone();
            cv::rectangle(ov2, cv::Point(w - 400, h - 15 - int(person_events.size()) * 25 - 10),
                          cv::Point(w, h), cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(ov2, 0.6, display, 0.4, 0, display);
            for (size_t ei = 0; ei < person_events.size(); ei++) {
                int ey = h - 15 - int(person_events.size() - 1 - ei) * 25;
                cv::Scalar ec = (person_events[ei].find("ENTRY") != std::string::npos)
                    ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::putText(display, person_events[ei], cv::Point(w - 390, ey),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, ec, 2);
            }
        }

        if (opts.save && writer.isOpened()) writer.write(display);

        if (!opts.headless) {
            cv::imshow("FastMatch", display);
            if (cv::waitKey(1) == 'q') break;
        }
    }

    for (auto& cap : cameras) cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyAllWindows();

    std::cout << "\nDone. Tracked " << persons.size() << " people over " << frame_num << " frames.";
    if (opts.line_enabled)
        std::cout << " Entries=" << entry_count << " Exits=" << exit_count;
    std::cout << std::endl;
    return 0;
}
