#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>

#include "scrfd.hpp"
#include "arcface.hpp"
#include "face_db.hpp"

namespace fs = std::filesystem;

// Threaded RTSP reader — always holds the latest frame
class StreamReader {
public:
    StreamReader(cv::VideoCapture& cap) : cap_(cap), running_(true) {
        thread_ = std::thread(&StreamReader::run, this);
    }
    ~StreamReader() { stop(); }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    bool read(cv::Mat& out) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (frame_.empty()) return false;
        out = frame_.clone();
        return true;
    }

private:
    void run() {
        cv::Mat tmp;
        while (running_) {
            if (cap_.read(tmp)) {
                std::lock_guard<std::mutex> lock(mtx_);
                frame_ = tmp;
            }
        }
    }

    cv::VideoCapture& cap_;
    std::atomic<bool> running_;
    std::thread thread_;
    std::mutex mtx_;
    cv::Mat frame_;
};

void print_usage(const char* prog) {
    std::cout << "Face Detection & Recognition Demo\n\n"
        << "Usage: " << prog << " [options]\n\n"
        << "Sources (default: webcam 0):\n"
        << "  --webcam [INDEX]       Webcam source (default 0)\n"
        << "  --rtsp URL [URL...]    RTSP stream(s)\n"
        << "  --video FILE           Video file\n"
        << "  --dir PATH             Directory of video files\n"
        << "\nOptions:\n"
        << "  --det-model PATH       SCRFD model (default: ../weights/det_10g.onnx)\n"
        << "  --rec-model PATH       ArcFace model (default: ../weights/w600k_mbf.onnx)\n"
        << "  --faces-dir PATH       Directory of face images to register\n"
        << "  --db PATH              Face database file (default: face_db.yaml)\n"
        << "  --threshold FLOAT      Similarity threshold (default: 0.4)\n"
        << "  --conf FLOAT           Detection confidence (default: 0.5)\n"
        << "  --max-width INT        Max processing width in pixels (default: 720, 0=no limit)\n"
        << "  --rebuild              Force rebuild face database\n"
        << "  --help                 Show this help\n"
        << "\nControls:\n"
        << "  r     Register face — freezes frame, select a face, type name\n"
        << "  d     Delete a registered face by name\n"
        << "  l     List all registered faces\n"
        << "  n/p   Next/previous source (multi-source mode)\n"
        << "  q     Quit\n";
}

// Draw numbered faces on frame for selection
void draw_numbered_faces(cv::Mat& frame, const std::vector<FaceDetection>& faces) {
    for (int i = 0; i < static_cast<int>(faces.size()); i++) {
        const auto& f = faces[i];
        int x1 = std::max(0, static_cast<int>(f.bbox.x));
        int y1 = std::max(0, static_cast<int>(f.bbox.y));
        int x2 = std::min(frame.cols, static_cast<int>(f.bbox.x + f.bbox.width));
        int y2 = std::min(frame.rows, static_cast<int>(f.bbox.y + f.bbox.height));

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        // Big number label
        std::string label = std::to_string(i + 1);
        int cx = (x1 + x2) / 2;
        int cy = (y1 + y2) / 2;
        cv::putText(frame, label, cv::Point(cx - 10, cy + 10),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 3);

        for (const auto& pt : f.landmarks)
            cv::circle(frame, cv::Point(int(pt.x), int(pt.y)), 2, cv::Scalar(0, 255, 255), -1);
    }
}

// Registration mode: freeze frame, detect faces, let user pick one and name it
bool register_face(cv::Mat& frozen_frame, SCRFD& detector, ArcFace& recognizer,
                   FaceDatabase& db, const std::string& db_path) {
    auto faces = detector.detect(frozen_frame);
    if (faces.empty()) {
        std::cout << "[Register] No faces detected in frame." << std::endl;
        return false;
    }

    // Draw numbered faces
    cv::Mat display = frozen_frame.clone();
    draw_numbered_faces(display, faces);

    // Instructions overlay
    std::string msg = "REGISTER: Press 1-" + std::to_string(faces.size()) + " to select face, ESC to cancel";
    cv::putText(display, msg, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 255), 2);
    cv::imshow("Face Recognition", display);

    // Wait for face selection
    int selected = -1;
    while (true) {
        int key = cv::waitKey(0) & 0xFF;
        if (key == 27) { // ESC
            std::cout << "[Register] Cancelled." << std::endl;
            return false;
        }
        int num = key - '0';
        if (num >= 1 && num <= static_cast<int>(faces.size())) {
            selected = num - 1;
            break;
        }
    }

    // Highlight selected face
    const auto& face = faces[selected];
    cv::Mat highlight = frozen_frame.clone();
    int x1 = std::max(0, static_cast<int>(face.bbox.x));
    int y1 = std::max(0, static_cast<int>(face.bbox.y));
    int x2 = std::min(frozen_frame.cols, static_cast<int>(face.bbox.x + face.bbox.width));
    int y2 = std::min(frozen_frame.rows, static_cast<int>(face.bbox.y + face.bbox.height));
    cv::rectangle(highlight, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255), 3);
    cv::putText(highlight, "Selected — enter name in terminal", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    cv::imshow("Face Recognition", highlight);
    cv::waitKey(1);

    // Get name from terminal
    std::cout << "[Register] Enter name for face #" << (selected + 1) << ": ";
    std::string name;
    std::getline(std::cin, name);

    if (name.empty()) {
        std::cout << "[Register] Empty name, cancelled." << std::endl;
        return false;
    }

    // Extract embedding and add to DB
    auto embedding = recognizer.get_embedding(frozen_frame, face.landmarks);
    db.add(name, embedding);
    db.build_index();
    db.save(db_path);

    std::cout << "[Register] Added '" << name << "' — DB now has " << db.size() << " faces" << std::endl;
    return true;
}

// Delete a face by name
bool delete_face(FaceDatabase& db, const std::string& db_path) {
    auto all_names = db.names();
    if (all_names.empty()) {
        std::cout << "[Delete] Database is empty." << std::endl;
        return false;
    }

    std::cout << "[Delete] Registered faces:" << std::endl;
    for (int i = 0; i < static_cast<int>(all_names.size()); i++)
        std::cout << "  " << (i + 1) << ". " << all_names[i] << std::endl;
    std::cout << "[Delete] Enter name to delete (or empty to cancel): ";

    std::string name;
    std::getline(std::cin, name);
    if (name.empty()) return false;

    if (db.remove(name)) {
        db.build_index();
        db.save(db_path);
        std::cout << "[Delete] Removed '" << name << "' — DB now has " << db.size() << " faces" << std::endl;
        return true;
    } else {
        std::cout << "[Delete] '" << name << "' not found." << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    enum SourceType { SRC_WEBCAM, SRC_VIDEO, SRC_RTSP, SRC_DIR };

    std::string det_model = "../weights/det_10g.onnx";
    std::string rec_model = "../weights/w600k_mbf.onnx";
    std::string faces_dir;
    std::string db_path = "face_db.yaml";
    std::string video_path;
    std::string dir_path;
    std::vector<std::string> rtsp_urls;
    float threshold = 0.4f;
    float conf = 0.5f;
    int webcam_idx = 0;
    int max_width = 720;
    SourceType source_type = SRC_WEBCAM;
    bool rebuild = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        else if (arg == "--webcam") {
            source_type = SRC_WEBCAM;
            if (i + 1 < argc && argv[i + 1][0] != '-') webcam_idx = std::stoi(argv[++i]);
        }
        else if (arg == "--video") { source_type = SRC_VIDEO; video_path = argv[++i]; }
        else if (arg == "--rtsp") {
            source_type = SRC_RTSP;
            while (i + 1 < argc && argv[i + 1][0] != '-')
                rtsp_urls.push_back(argv[++i]);
        }
        else if (arg == "--dir") { source_type = SRC_DIR; dir_path = argv[++i]; }
        else if (arg == "--det-model") det_model = argv[++i];
        else if (arg == "--rec-model") rec_model = argv[++i];
        else if (arg == "--faces-dir") faces_dir = argv[++i];
        else if (arg == "--db") db_path = argv[++i];
        else if (arg == "--threshold") threshold = std::stof(argv[++i]);
        else if (arg == "--conf") conf = std::stof(argv[++i]);
        else if (arg == "--max-width") max_width = std::stoi(argv[++i]);
        else if (arg == "--rebuild") rebuild = true;
    }

    // Load models
    std::cout << "Loading SCRFD: " << det_model << std::endl;
    SCRFD detector(det_model, {640, 640}, conf);

    std::cout << "Loading ArcFace: " << rec_model << std::endl;
    ArcFace recognizer(rec_model);

    // Face database
    FaceDatabase db;
    bool loaded = false;
    if (!rebuild) loaded = db.load(db_path);

    if (loaded) {
        std::cout << "Loaded face database: " << db.size() << " faces" << std::endl;
        auto all_names = db.names();
        for (const auto& n : all_names) std::cout << "  - " << n << std::endl;
    }

    // Build/update from faces directory
    if (!faces_dir.empty() && (rebuild || !loaded)) {
        std::cout << "Building face database from: " << faces_dir << std::endl;
        std::vector<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp"};

        for (const auto& entry : fs::directory_iterator(faces_dir)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(exts.begin(), exts.end(), ext) == exts.end()) continue;

            std::string name = entry.path().stem().string();
            cv::Mat img = cv::imread(entry.path().string());
            if (img.empty()) { std::cerr << "  Skip (unreadable): " << name << std::endl; continue; }

            auto faces = detector.detect(img, 1);
            if (faces.empty()) { std::cerr << "  Skip (no face): " << name << std::endl; continue; }

            auto embedding = recognizer.get_embedding(img, faces[0].landmarks);
            db.add(name, embedding);
            std::cout << "  Added: " << name << std::endl;
        }
        db.build_index();
        db.save(db_path);
        std::cout << "Database saved: " << db.size() << " faces"
                  << (FaceDatabase::faiss_available() ? " [FAISS]" : " [linear]")
                  << std::endl;
    }

    // Open video source(s)
    std::vector<cv::VideoCapture> caps;
    std::vector<std::string> source_labels;

    switch (source_type) {
    case SRC_WEBCAM: {
        cv::VideoCapture cap(webcam_idx);
        if (!cap.isOpened()) { std::cerr << "Cannot open webcam " << webcam_idx << std::endl; return 1; }
        caps.push_back(std::move(cap));
        source_labels.push_back("Webcam " + std::to_string(webcam_idx));
        break;
    }
    case SRC_VIDEO: {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) { std::cerr << "Cannot open video: " << video_path << std::endl; return 1; }
        caps.push_back(std::move(cap));
        source_labels.push_back("Video: " + video_path);
        break;
    }
    case SRC_RTSP: {
        if (rtsp_urls.empty()) { std::cerr << "No RTSP URLs provided" << std::endl; return 1; }
        for (const auto& url : rtsp_urls) {
            cv::VideoCapture cap;
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            cap.open(url, cv::CAP_FFMPEG, {
                cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            });
            if (!cap.isOpened()) { std::cerr << "Cannot open RTSP: " << url << std::endl; continue; }
            caps.push_back(std::move(cap));
            source_labels.push_back("RTSP: " + url);
        }
        if (caps.empty()) { std::cerr << "No RTSP streams opened" << std::endl; return 1; }
        break;
    }
    case SRC_DIR: {
        std::vector<std::string> vid_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv"};
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(vid_exts.begin(), vid_exts.end(), ext) == vid_exts.end()) continue;
            cv::VideoCapture cap(entry.path().string());
            if (!cap.isOpened()) { std::cerr << "Skip: " << entry.path().filename() << std::endl; continue; }
            caps.push_back(std::move(cap));
            source_labels.push_back(entry.path().filename().string());
        }
        if (caps.empty()) { std::cerr << "No videos found in: " << dir_path << std::endl; return 1; }
        break;
    }
    }

    for (const auto& lbl : source_labels) std::cout << "Source: " << lbl << std::endl;

    // Start threaded readers for RTSP sources
    std::vector<std::unique_ptr<StreamReader>> readers;
    if (source_type == SRC_RTSP) {
        for (auto& cap : caps)
            readers.push_back(std::make_unique<StreamReader>(cap));
    }

    std::cout << "\nControls: [r] register  [d] delete  [l] list  [q] quit\n" << std::endl;

    // Random colors per name
    std::unordered_map<std::string, cv::Scalar> colors;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> cd(50, 255);

    int active_src = 0;
    cv::Mat frame, proc_frame;
    int frame_num = 0;
    while (true) {
        bool got_frame = false;

        if (source_type == SRC_RTSP) {
            // Threaded reader always has latest frame
            if (active_src < static_cast<int>(readers.size()))
                got_frame = readers[active_src]->read(frame);
        } else {
            for (size_t s = 0; s < caps.size(); s++) {
                cv::Mat tmp;
                if (caps[s].read(tmp)) {
                    if (static_cast<int>(s) == active_src) { frame = tmp; got_frame = true; }
                }
            }
        }
        if (!got_frame) {
            if (source_type == SRC_RTSP) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
            else break;
        }
        if (frame.empty()) continue;
        frame_num++;

        // Downscale large frames for faster processing
        float scale = 1.0f;
        if (max_width > 0 && frame.cols > max_width) {
            scale = static_cast<float>(max_width) / frame.cols;
            cv::resize(frame, proc_frame, cv::Size(), scale, scale);
        } else {
            proc_frame = frame;
        }

        auto t0 = std::chrono::steady_clock::now();

        // Detect on (possibly downscaled) proc_frame
        auto faces = detector.detect(proc_frame);

        // Draw on proc_frame (which is display-sized)
        for (const auto& face : faces) {
            int x1 = std::max(0, static_cast<int>(face.bbox.x));
            int y1 = std::max(0, static_cast<int>(face.bbox.y));
            int x2 = std::min(proc_frame.cols, static_cast<int>(face.bbox.x + face.bbox.width));
            int y2 = std::min(proc_frame.rows, static_cast<int>(face.bbox.y + face.bbox.height));

            if (db.size() > 0) {
                auto embedding = recognizer.get_embedding(proc_frame, face.landmarks);
                auto [name, sim] = db.search(embedding, threshold);

                if (name != "Unknown") {
                    if (!colors.count(name))
                        colors[name] = cv::Scalar(cd(rng), cd(rng), cd(rng));
                    auto& c = colors[name];
                    cv::rectangle(proc_frame, cv::Point(x1, y1), cv::Point(x2, y2), c, 2);
                    char label[128];
                    snprintf(label, sizeof(label), "%s: %.2f", name.c_str(), sim);
                    cv::putText(proc_frame, label, cv::Point(x1, y1 - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, c, 2);
                } else {
                    cv::rectangle(proc_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                                  cv::Scalar(0, 0, 255), 2);
                    cv::putText(proc_frame, "Unknown", cv::Point(x1, y1 - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1);
                }
            } else {
                cv::rectangle(proc_frame, cv::Point(x1, y1), cv::Point(x2, y2),
                              cv::Scalar(0, 255, 0), 2);
            }

            // Draw landmarks
            for (const auto& pt : face.landmarks)
                cv::circle(proc_frame, cv::Point(int(pt.x), int(pt.y)), 2, cv::Scalar(0, 255, 255), -1);
        }

        auto t1 = std::chrono::steady_clock::now();
        double fps = 1.0 / (std::chrono::duration<double>(t1 - t0).count() + 1e-9);

        char hud[256];
        if (caps.size() > 1) {
            snprintf(hud, sizeof(hud), "FPS: %.1f  Faces: %zu  DB: %d  Src: %d/%zu  %dx%d [n/p]switch [r]eg [d]el [l]ist [q]uit",
                     fps, faces.size(), db.size(), active_src + 1, caps.size(), frame.cols, frame.rows);
        } else {
            snprintf(hud, sizeof(hud), "FPS: %.1f  Faces: %zu  DB: %d  %dx%d  [r]egister [d]elete [l]ist [q]uit",
                     fps, faces.size(), db.size(), frame.cols, frame.rows);
        }
        cv::putText(proc_frame, hud, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);

        cv::imshow("Face Recognition", proc_frame);
        int key = cv::waitKey(1) & 0xFF;

        if (key == 'q') break;
        else if (key == 'r') {
            // Freeze on current frame for registration
            register_face(proc_frame, detector, recognizer, db, db_path);
        }
        else if (key == 'd') {
            delete_face(db, db_path);
        }
        else if (key == 'l') {
            auto all_names = db.names();
            std::cout << "\n[DB] " << db.size() << " faces registered";
            if (all_names.empty()) { std::cout << " (empty)" << std::endl; }
            else {
                std::cout << ":" << std::endl;
                for (const auto& n : all_names) std::cout << "  - " << n << std::endl;
            }
        }
        else if (key == 'n' && caps.size() > 1) {
            active_src = (active_src + 1) % static_cast<int>(caps.size());
            std::cout << "[Source] Switched to " << source_labels[active_src] << std::endl;
        }
        else if (key == 'p' && caps.size() > 1) {
            active_src = (active_src - 1 + static_cast<int>(caps.size())) % static_cast<int>(caps.size());
            std::cout << "[Source] Switched to " << source_labels[active_src] << std::endl;
        }
    }

    // Stop threaded readers before releasing captures
    readers.clear();
    for (auto& c : caps) c.release();
    cv::destroyAllWindows();
    std::cout << "\nProcessed " << frame_num << " frames." << std::endl;
    return 0;
}
