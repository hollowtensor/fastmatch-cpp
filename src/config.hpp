#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace fs = std::filesystem;

// Model/threshold config — loaded from YAML
struct ModelConfig {
    std::string detection_model;
    std::string detection_classes;
    std::string reid_model;
    std::string device = "cpu";

    float detection_threshold = 0.3f;
    float reid_threshold = 0.42f;
    int max_gallery = 512;

    static ModelConfig load(const std::string& path) {
        auto base_dir = fs::path(path).parent_path();
        YAML::Node y = YAML::LoadFile(path);
        ModelConfig m;

        auto resolve = [&](const std::string& p) -> std::string {
            fs::path fp(p);
            if (fp.is_absolute()) return p;
            return (base_dir / fp).string();
        };

        m.detection_model = resolve(y["object_detection_model_path"].as<std::string>("./pretrained_models/yolov4-tiny.onnx"));
        m.detection_classes = resolve(y["object_detection_classes_path"].as<std::string>("./pretrained_models/coco.names"));
        m.reid_model = resolve(y["feature_extraction_model_path"].as<std::string>("./pretrained_models/osnet_ain_x1_0_M.onnx"));
        m.device = y["inference_model_device"].as<std::string>("cpu");
        m.detection_threshold = y["object_detection_threshold"].as<float>(0.3f);
        m.reid_threshold = y["feature_extraction_threshold"].as<float>(0.42f);
        m.max_gallery = y["max_gallery_set_each_person"].as<int>(512);

        return m;
    }
};

// Runtime options — from CLI args
struct RuntimeOpts {
    // Source (exactly one must be set)
    enum SourceType { NONE, WEBCAM, RTSP, VIDEO_FILE, VIDEO_DIR };
    SourceType source_type = NONE;
    int webcam_index = 0;
    std::vector<std::string> rtsp_urls;
    std::string video_path;  // file or directory

    // Display
    int width = 1280;
    int height = 720;
    float display_scale = 1.0f;
    bool headless = false;

    // Output
    bool save = false;
    std::string output_path = "./output.avi";
    float output_fps = 30.0f;

    // Zone (optional)
    bool zone_enabled = false;
    std::vector<cv::Point> zone_points;

    // Entry line (optional)
    bool line_enabled = false;
    cv::Point line_p1, line_p2;
    int entry_sign = -1;

    // Calibration
    bool calibrate = false;

    // Config file for models
    std::string config_path;

    static void print_usage(const char* prog) {
        std::cout << "FastMatch - Multi-Camera People Tracking\n\n"
            << "Usage: " << prog << " <source> [options]\n\n"
            << "Sources (pick one):\n"
            << "  --webcam [INDEX]         Use webcam (default index 0)\n"
            << "  --rtsp URL [URL...]      RTSP stream(s)\n"
            << "  --video FILE             Video file\n"
            << "  --dir DIRECTORY          All videos in directory\n\n"
            << "Options:\n"
            << "  --config PATH            Model config YAML (default: ../config.yaml)\n"
            << "  --size WxH               Frame size (default: 1280x720)\n"
            << "  --scale FLOAT            Display scale (default: 1.0)\n"
            << "  --headless               No display window\n"
            << "  --save PATH              Save output video to PATH\n"
            << "  --fps FLOAT              Output video FPS (default: 30)\n"
            << "  --calibrate              Interactive zone + line setup\n"
            << "  --zone X1,Y1,X2,Y2,...   Detection zone polygon\n"
            << "  --line X1,Y1,X2,Y2,SIGN Entry/exit line\n"
            << "  --help                   Show this help\n\n"
            << "Examples:\n"
            << "  " << prog << " --webcam\n"
            << "  " << prog << " --webcam 1 --size 640x480\n"
            << "  " << prog << " --video traffic.mp4 --save out.avi\n"
            << "  " << prog << " --rtsp rtsp://user:pass@192.168.0.1/stream1\n"
            << "  " << prog << " --dir ./videos --headless --save result.avi\n"
            << "  " << prog << " --video cam.mp4 --zone 670,372,381,402,505,698,902,606\n"
            << "  " << prog << " --video cam.mp4 --line 621,379,848,620,-1\n";
    }

    static RuntimeOpts parse(int argc, char** argv) {
        RuntimeOpts o;

        o.config_path = "../config.yaml";

        if (argc < 2) {
            print_usage(argv[0]);
            std::exit(0);
        }

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                std::exit(0);
            }
            else if (arg == "--webcam") {
                o.source_type = WEBCAM;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    o.webcam_index = std::stoi(argv[++i]);
                }
            }
            else if (arg == "--rtsp") {
                o.source_type = RTSP;
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    o.rtsp_urls.push_back(argv[++i]);
                }
                if (o.rtsp_urls.empty()) {
                    std::cerr << "Error: --rtsp requires at least one URL\n";
                    std::exit(1);
                }
            }
            else if (arg == "--video") {
                if (i + 1 >= argc) { std::cerr << "Error: --video requires a path\n"; std::exit(1); }
                o.source_type = VIDEO_FILE;
                o.video_path = argv[++i];
            }
            else if (arg == "--dir") {
                if (i + 1 >= argc) { std::cerr << "Error: --dir requires a path\n"; std::exit(1); }
                o.source_type = VIDEO_DIR;
                o.video_path = argv[++i];
            }
            else if (arg == "--config") {
                if (i + 1 >= argc) { std::cerr << "Error: --config requires a path\n"; std::exit(1); }
                o.config_path = argv[++i];
            }
            else if (arg == "--size") {
                if (i + 1 >= argc) { std::cerr << "Error: --size requires WxH\n"; std::exit(1); }
                std::string s = argv[++i];
                auto x = s.find('x');
                if (x == std::string::npos) { std::cerr << "Error: --size format is WxH (e.g. 1280x720)\n"; std::exit(1); }
                o.width = std::stoi(s.substr(0, x));
                o.height = std::stoi(s.substr(x + 1));
            }
            else if (arg == "--scale") {
                if (i + 1 >= argc) { std::cerr << "Error: --scale requires a value\n"; std::exit(1); }
                o.display_scale = std::stof(argv[++i]);
            }
            else if (arg == "--headless") {
                o.headless = true;
            }
            else if (arg == "--calibrate") {
                o.calibrate = true;
            }
            else if (arg == "--save") {
                o.save = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    o.output_path = argv[++i];
                }
            }
            else if (arg == "--fps") {
                if (i + 1 >= argc) { std::cerr << "Error: --fps requires a value\n"; std::exit(1); }
                o.output_fps = std::stof(argv[++i]);
            }
            else if (arg == "--zone") {
                if (i + 1 >= argc) { std::cerr << "Error: --zone requires coordinates\n"; std::exit(1); }
                o.zone_enabled = true;
                std::string coords = argv[++i];
                // Parse comma-separated: x1,y1,x2,y2,...
                std::stringstream ss(coords);
                std::string token;
                std::vector<int> vals;
                while (std::getline(ss, token, ',')) vals.push_back(std::stoi(token));
                if (vals.size() < 6 || vals.size() % 2 != 0) {
                    std::cerr << "Error: --zone needs at least 3 points (6 values)\n"; std::exit(1);
                }
                for (size_t j = 0; j < vals.size(); j += 2)
                    o.zone_points.emplace_back(vals[j], vals[j + 1]);
            }
            else if (arg == "--line") {
                if (i + 1 >= argc) { std::cerr << "Error: --line requires coordinates\n"; std::exit(1); }
                o.line_enabled = true;
                std::string coords = argv[++i];
                std::stringstream ss(coords);
                std::string token;
                std::vector<int> vals;
                while (std::getline(ss, token, ',')) vals.push_back(std::stoi(token));
                if (vals.size() != 5) {
                    std::cerr << "Error: --line format: X1,Y1,X2,Y2,SIGN\n"; std::exit(1);
                }
                o.line_p1 = cv::Point(vals[0], vals[1]);
                o.line_p2 = cv::Point(vals[2], vals[3]);
                o.entry_sign = vals[4];
            }
            else {
                std::cerr << "Unknown option: " << arg << "\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        }

        if (o.source_type == NONE) {
            std::cerr << "Error: No source specified. Use --webcam, --rtsp, --video, or --dir\n";
            std::exit(1);
        }

        return o;
    }
};
