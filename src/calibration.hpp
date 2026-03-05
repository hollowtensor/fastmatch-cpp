#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <opencv2/opencv.hpp>

struct CalibrationResult {
    bool has_zone = false;
    std::vector<cv::Point> zone_points;

    bool has_line = false;
    cv::Point line_p1, line_p2;
    int entry_sign = -1;

    // Save to file
    void save(const std::string& path) const {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        if (has_zone) {
            fs << "zone_points" << "[";
            for (const auto& p : zone_points)
                fs << "{:" << "x" << p.x << "y" << p.y << "}";
            fs << "]";
        }
        if (has_line) {
            fs << "line_p1" << "{:" << "x" << line_p1.x << "y" << line_p1.y << "}";
            fs << "line_p2" << "{:" << "x" << line_p2.x << "y" << line_p2.y << "}";
            fs << "entry_sign" << entry_sign;
        }
        fs.release();
        std::cout << "Calibration saved to: " << path << std::endl;
    }

    // Load from file, returns false if file doesn't exist
    static bool load(const std::string& path, CalibrationResult& out) {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;

        out = {};
        if (!fs["zone_points"].empty()) {
            cv::FileNode zn = fs["zone_points"];
            for (const auto& pt : zn)
                out.zone_points.emplace_back((int)pt["x"], (int)pt["y"]);
            out.has_zone = !out.zone_points.empty();
        }
        if (!fs["line_p1"].empty()) {
            out.line_p1 = {(int)fs["line_p1"]["x"], (int)fs["line_p1"]["y"]};
            out.line_p2 = {(int)fs["line_p2"]["x"], (int)fs["line_p2"]["y"]};
            out.entry_sign = (int)fs["entry_sign"];
            out.has_line = true;
        }
        fs.release();
        std::cout << "Loaded calibration from: " << path << std::endl;
        return true;
    }
};

namespace calibration {

// Shared mouse state
struct MouseState {
    std::vector<cv::Point> points;
    cv::Point hover{-1, -1};
    bool closed = false;
    bool clicked = false;
};

inline void zone_mouse_cb(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* s = static_cast<MouseState*>(userdata);
    s->hover = {x, y};
    if (event == cv::EVENT_LBUTTONDOWN && !s->closed) {
        if (s->points.size() >= 3) {
            double d = std::hypot(x - s->points[0].x, y - s->points[0].y);
            if (d < 15) { s->closed = true; return; }
        }
        s->points.emplace_back(x, y);
    }
}

inline void line_mouse_cb(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* s = static_cast<MouseState*>(userdata);
    s->hover = {x, y};
    if (event == cv::EVENT_LBUTTONDOWN) {
        if (s->points.size() < 2)
            s->points.emplace_back(x, y);
        else
            s->clicked = true;
    }
}

// Phase 1: Zone calibration
// Returns empty vector if user skips (press 's')
inline std::vector<cv::Point> calibrate_zone(const cv::Mat& frame) {
    std::cout << "\n=== ZONE CALIBRATION ===\n"
              << "Click points to draw detection zone polygon.\n"
              << "Click near first point to close | c=close | r=reset | s=skip | q=quit\n" << std::endl;

    MouseState state;
    const std::string win = "Calibrate Zone";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win, zone_mouse_cb, &state);

    while (true) {
        cv::Mat disp = frame.clone();

        // Draw filled polygon preview
        if (state.points.size() >= 3) {
            cv::Mat overlay = disp.clone();
            std::vector<cv::Point> pts(state.points);
            if (state.closed) {
                cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(0, 200, 0));
                cv::addWeighted(overlay, 0.3, disp, 0.7, 0, disp);
                cv::polylines(disp, std::vector<std::vector<cv::Point>>{pts}, true, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::polylines(overlay, std::vector<std::vector<cv::Point>>{pts}, false, cv::Scalar(0, 255, 0), 2);
                cv::addWeighted(overlay, 0.3, disp, 0.7, 0, disp);
                // Preview line to cursor
                if (state.hover.x >= 0)
                    cv::line(disp, state.points.back(), state.hover, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        } else if (state.points.size() >= 1 && state.hover.x >= 0) {
            cv::line(disp, state.points.back(), state.hover, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        // Draw points
        for (size_t i = 0; i < state.points.size(); i++) {
            cv::Scalar c = (i == 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::circle(disp, state.points[i], 6, c, -1);
            cv::putText(disp, std::to_string(i + 1),
                cv::Point(state.points[i].x + 8, state.points[i].y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, c, 1);
        }
        // Lines between points
        for (size_t i = 1; i < state.points.size(); i++)
            cv::line(disp, state.points[i - 1], state.points[i], cv::Scalar(0, 255, 0), 2);

        // Instructions
        char info[128];
        if (state.closed)
            snprintf(info, sizeof(info), "Points: %zu | CLOSED - press ENTER to confirm", state.points.size());
        else
            snprintf(info, sizeof(info), "Points: %zu | Click to add, c=close", state.points.size());
        cv::putText(disp, info, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(disp, "r=reset  s=skip  q=quit  c=close  ENTER=confirm",
            cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imshow(win, disp);
        int key = cv::waitKey(16) & 0xFF;

        if (key == 'q') {
            cv::destroyWindow(win);
            std::exit(0);
        }
        if (key == 'r') {
            state.points.clear();
            state.closed = false;
        }
        if (key == 's') {
            cv::destroyWindow(win);
            return {};
        }
        if (key == 'c' && state.points.size() >= 3)
            state.closed = true;
        if ((key == 13 || key == 10) && state.closed) // ENTER
            break;
    }

    cv::destroyWindow(win);
    std::cout << "Zone: " << state.points.size() << " vertices" << std::endl;
    return state.points;
}

// Phase 2: Line calibration
// Returns false if user skips
inline bool calibrate_line(const cv::Mat& frame,
                           const std::vector<cv::Point>& zone,
                           cv::Point& out_p1, cv::Point& out_p2, int& out_sign) {
    std::cout << "\n=== LINE CALIBRATION ===\n"
              << "Step 1: Click TWO points for the entry/exit line.\n"
              << "r=reset | s=skip | q=quit\n" << std::endl;

    MouseState state;
    const std::string win = "Calibrate Line";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win, line_mouse_cb, &state);

    // Step 1: draw line
    while (true) {
        cv::Mat disp = frame.clone();

        // Draw zone if present
        if (!zone.empty()) {
            cv::Mat zo = disp.clone();
            cv::fillPoly(zo, std::vector<std::vector<cv::Point>>{zone}, cv::Scalar(0, 200, 0));
            cv::addWeighted(zo, 0.2, disp, 0.8, 0, disp);
            cv::polylines(disp, std::vector<std::vector<cv::Point>>{zone}, true, cv::Scalar(0, 255, 0), 2);
        }

        // Draw clicked points
        for (auto& pt : state.points)
            cv::circle(disp, pt, 6, cv::Scalar(0, 255, 0), -1);

        if (state.points.size() == 1 && state.hover.x >= 0)
            cv::line(disp, state.points[0], state.hover, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        if (state.points.size() == 2)
            cv::line(disp, state.points[0], state.points[1], cv::Scalar(0, 255, 255), 2);

        char info[128];
        snprintf(info, sizeof(info), "Line points: %zu/2%s",
            state.points.size(), state.points.size() == 2 ? " | Press ENTER to confirm" : "");
        cv::putText(disp, info, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(disp, "r=reset  s=skip  q=quit",
            cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imshow(win, disp);
        int key = cv::waitKey(16) & 0xFF;

        if (key == 'q') { cv::destroyWindow(win); std::exit(0); }
        if (key == 'r') { state.points.clear(); }
        if (key == 's') { cv::destroyWindow(win); return false; }
        if ((key == 13 || key == 10) && state.points.size() == 2) break;
    }

    out_p1 = state.points[0];
    out_p2 = state.points[1];
    cv::destroyWindow(win);

    // Step 2: click entry side
    std::cout << "Step 2: Click on the ENTRY side of the line.\n" << std::endl;

    MouseState side_state;
    const std::string win2 = "Click ENTRY side";
    cv::namedWindow(win2, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(win2, line_mouse_cb, &side_state);

    while (!side_state.clicked) {
        cv::Mat disp = frame.clone();

        if (!zone.empty()) {
            cv::Mat zo = disp.clone();
            cv::fillPoly(zo, std::vector<std::vector<cv::Point>>{zone}, cv::Scalar(0, 200, 0));
            cv::addWeighted(zo, 0.2, disp, 0.8, 0, disp);
            cv::polylines(disp, std::vector<std::vector<cv::Point>>{zone}, true, cv::Scalar(0, 255, 0), 2);
        }

        cv::line(disp, out_p1, out_p2, cv::Scalar(0, 255, 255), 2);

        // Show side preview on hover
        if (side_state.hover.x >= 0) {
            float sv = static_cast<float>(
                (out_p2.x - out_p1.x) * (side_state.hover.y - out_p1.y) -
                (out_p2.y - out_p1.y) * (side_state.hover.x - out_p1.x));
            cv::Scalar hc = (sv > 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 100, 255);
            cv::circle(disp, side_state.hover, 12, hc, 2);
        }

        cv::putText(disp, "CLICK on the ENTRY side of the line",
            cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(disp, "(the side people come FROM when entering)",
            cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        cv::imshow(win2, disp);
        int key = cv::waitKey(16) & 0xFF;
        if (key == 'q') { cv::destroyWindow(win2); std::exit(0); }
    }

    cv::Point click = side_state.points[0];
    float sv = static_cast<float>(
        (out_p2.x - out_p1.x) * (click.y - out_p1.y) -
        (out_p2.y - out_p1.y) * (click.x - out_p1.x));
    out_sign = (sv > 0) ? 1 : -1;

    cv::destroyWindow(win2);
    std::cout << "Line: (" << out_p1.x << "," << out_p1.y << ")->(" << out_p2.x << "," << out_p2.y
              << ") entry_sign=" << out_sign << std::endl;
    return true;
}

// Full calibration flow: zone then line
inline CalibrationResult run_calibration(cv::VideoCapture& cap, cv::Size cam_size) {
    // Let user pick a good frame — live preview, press SPACE/ENTER to freeze
    std::cout << "\n=== FRAME SELECTION ===\n"
              << "Live preview. Press SPACE or ENTER when the frame looks good.\n"
              << "q=quit\n" << std::endl;

    cv::Mat frame;
    const std::string pick_win = "Select Frame";
    cv::namedWindow(pick_win, cv::WINDOW_AUTOSIZE);

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "ERROR: Cannot read frame." << std::endl;
            std::exit(1);
        }
        cv::resize(frame, frame, cam_size, 0, 0, cv::INTER_CUBIC);

        cv::Mat disp = frame.clone();
        cv::putText(disp, "Press SPACE/ENTER to use this frame for calibration",
            cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::imshow(pick_win, disp);

        int key = cv::waitKey(30) & 0xFF;
        if (key == ' ' || key == 13 || key == 10) break;
        if (key == 'q') { cv::destroyWindow(pick_win); std::exit(0); }
    }
    cv::destroyWindow(pick_win);

    // Reset capture to start so tracking begins from frame 0
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    CalibrationResult result;

    // Phase 1: Zone
    result.zone_points = calibrate_zone(frame);
    result.has_zone = !result.zone_points.empty();

    // Phase 2: Line
    result.has_line = calibrate_line(frame, result.zone_points,
                                     result.line_p1, result.line_p2, result.entry_sign);

    // Save for next run
    result.save("calibration.yaml");

    return result;
}

} // namespace calibration
