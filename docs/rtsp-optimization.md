# RTSP Stream Optimization

## Problem

RTSP streams run significantly slower than webcam sources despite using the same detection/recognition pipeline. Two root causes:

1. **Frame buffer bloat** — OpenCV's `VideoCapture` internally buffers multiple RTSP frames. By the time the main loop calls `read()`, it decodes stale frames sequentially, causing the pipeline to fall behind real-time.

2. **High resolution** — RTSP cameras typically stream at 1080p or higher, while webcams default to 480p/720p. Every frame takes longer to decode, resize (inside SCRFD letterbox), and process through ArcFace alignment.

## Solution

### Threaded Stream Reader

A dedicated background thread per RTSP source continuously calls `read()` on the `VideoCapture`, keeping only the latest frame in memory. The main processing loop fetches the most recent frame instantly without blocking on network I/O or decoding delays.

```
[RTSP Stream] --> [Background Thread: continuous read()] --> [Latest Frame Buffer]
                                                                    |
[Main Loop] <-- instant read -----------------------------------------
```

This decouples frame acquisition from frame processing. If processing takes 50ms but the stream delivers at 30fps (33ms), without threading the pipeline falls behind by ~17ms per frame, accumulating latency. With threading, we always skip to the latest available frame.

### OpenCV Capture Settings

```cpp
cap.set(cv::CAP_PROP_BUFFERSIZE, 1);      // Minimize internal buffer
cap.open(url, cv::CAP_FFMPEG, {
    cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,  // Connection timeout
    cv::CAP_PROP_READ_TIMEOUT_MSEC, 5000,  // Read timeout
});
```

- `CAP_PROP_BUFFERSIZE = 1` reduces OpenCV's internal frame queue
- `CAP_FFMPEG` backend is generally faster than GStreamer for RTSP
- Timeouts prevent hangs on network issues

### Auto-Downscale

Large frames are automatically downscaled before detection and recognition:

```
--max-width 720    (default) — 1920x1080 becomes 720x405
--max-width 1080   — keep up to 1080p
--max-width 0      — no downscaling (full resolution)
```

The SCRFD detector internally resizes to 640x640 anyway, so feeding it a 1080p frame wastes time on the initial resize. Downscaling first means:
- Faster `cv::resize` inside SCRFD (smaller source)
- Faster ArcFace face crop extraction
- Faster display rendering
- Minimal accuracy loss (faces are still well-resolved at 720p)

## Performance Comparison

| Source | Typical Resolution | Without Fixes | With Fixes |
|--------|-------------------|---------------|------------|
| Webcam | 640x480 | ~20-30 FPS | ~20-30 FPS |
| RTSP 1080p | 1920x1080 | ~3-5 FPS | ~15-25 FPS |
| RTSP 4K | 3840x2160 | <2 FPS | ~10-15 FPS |

*Actual FPS depends on hardware, number of faces, and network conditions.*

## Usage

```bash
# Default (720p processing, threaded reader)
./face_demo --rtsp "rtsp://user:pass@192.168.0.1:554/stream1"

# Higher quality
./face_demo --rtsp "rtsp://..." --max-width 1080

# Full resolution (slowest, best quality)
./face_demo --rtsp "rtsp://..." --max-width 0

# Multiple streams
./face_demo --rtsp "rtsp://cam1/stream" "rtsp://cam2/stream"
```

## Architecture

```
Per RTSP source:
  VideoCapture ---> StreamReader (background thread)
                         |
                    [mutex-protected latest frame]
                         |
Main loop:          read() ---> downscale ---> SCRFD ---> ArcFace ---> display
```

The `StreamReader` class:
- Runs `cap.read()` in a tight loop on a separate thread
- Stores the latest frame behind a mutex
- Main thread calls `reader.read(frame)` which returns immediately with the latest frame
- Automatically stopped and joined on destruction
