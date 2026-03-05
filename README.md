# FastMatch - Multi-Camera People Tracking (C++)

Fast multi-camera people tracking using YOLOv4-tiny (detection) and OSNet (re-identification) via ONNX Runtime. Detects people across multiple camera feeds and assigns consistent IDs using cosine similarity on feature embeddings.

## Dependencies

- **OpenCV 4.x**
- **ONNX Runtime**
- **yaml-cpp**
- **CMake 3.14+**

### macOS (Homebrew)

```bash
brew install opencv onnxruntime yaml-cpp cmake
```

### Ubuntu/Debian

```bash
sudo apt install libopencv-dev libyaml-cpp-dev cmake
# ONNX Runtime: download from https://github.com/microsoft/onnxruntime/releases
# Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime when running cmake
```

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Download Models

```bash
./download_models.sh
```

Or manually place these in `pretrained_models/`:
- `yolov4-tiny.onnx`
- `osnet_ain_x1_0_M.onnx`
- `coco.names`

## Usage

```bash
# Webcam
./build/fastmatch --webcam

# Video file
./build/fastmatch --video path/to/video.mp4

# RTSP stream(s)
./build/fastmatch --rtsp rtsp://user:pass@192.168.0.1/stream1

# Multiple RTSP streams
./build/fastmatch --rtsp rtsp://cam1/stream rtsp://cam2/stream

# Directory of video files
./build/fastmatch --dir ./videos
```

### Interactive Calibration

Draw a detection zone and entry/exit line on a live preview:

```bash
./build/fastmatch --webcam --calibrate
```

1. **Frame selection** - live feed plays, press **SPACE/ENTER** to freeze a good frame
2. **Zone** - click polygon vertices, press **c** to close, **ENTER** to confirm (or **s** to skip)
3. **Line** - click two points for the entry/exit line, **ENTER** to confirm
4. **Entry side** - click on the side people enter from

Calibration saves to `calibration.yaml` and auto-loads on the next run.

### Options

| Flag | Description |
|------|-------------|
| `--webcam [INDEX]` | Use webcam (default index 0) |
| `--rtsp URL [URL...]` | RTSP stream(s) |
| `--video FILE` | Video file |
| `--dir DIRECTORY` | All videos in a directory |
| `--calibrate` | Interactive zone + line setup |
| `--config PATH` | Model config YAML (default: `../config.yaml`) |
| `--size WxH` | Frame size (default: `1280x720`) |
| `--scale FLOAT` | Display scale (default: `1.0`) |
| `--headless` | No display window |
| `--save PATH` | Save output video |
| `--fps FLOAT` | Output video FPS (default: `30`) |
| `--zone X1,Y1,X2,Y2,...` | Detection zone polygon (CLI override) |
| `--line X1,Y1,X2,Y2,SIGN` | Entry/exit line (CLI override) |

### Examples

```bash
# Webcam with custom resolution
./build/fastmatch --webcam --size 640x480

# Video with zone filter (no calibration needed)
./build/fastmatch --video cam.mp4 --zone 670,372,381,402,505,698,902,606

# Video with entry/exit line
./build/fastmatch --video cam.mp4 --line 621,379,848,620,-1

# Headless processing, save output
./build/fastmatch --video input.mp4 --headless --save output.avi

# RTSP with calibration
./build/fastmatch --rtsp rtsp://192.168.0.100/stream --calibrate
```

## Configuration

`config.yaml` holds model paths and thresholds:

```yaml
object_detection_model_path: ./pretrained_models/yolov4-tiny.onnx
object_detection_classes_path: ./pretrained_models/coco.names
object_detection_threshold: 0.3
feature_extraction_model_path: ./pretrained_models/osnet_ain_x1_0_M.onnx
feature_extraction_threshold: 0.40
inference_model_device: cpu    # or "cuda"
max_gallery_set_each_person: 512
```

## Architecture

```
src/
├── main.cpp                 # Pipeline: capture → detect → re-id → track → display
├── config.hpp               # CLI arg parser + YAML model config loader
├── calibration.hpp          # Interactive zone/line calibration with persistence
├── object_detection.hpp/cpp # YOLOv4-tiny ONNX wrapper with NMS
├── feature_extraction.hpp/cpp # OSNet ONNX wrapper (512-d embeddings)
└── helpers.hpp/cpp          # Cosine distance, coord scaling, image stacking
```

## Controls

- **q** - quit
- During calibration: **r** = reset, **s** = skip, **c** = close polygon, **ENTER** = confirm
