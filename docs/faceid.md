# FaceID Module

Real-time face detection and recognition module for FastMatch. Detects faces using SCRFD, extracts 512-dimensional embeddings with ArcFace, and matches against a persistent face database. Optional FAISS backend for fast similarity search.

## Architecture

```
src/faceid/
  scrfd.hpp / scrfd.cpp         Face detection (SCRFD)
  arcface.hpp / arcface.cpp     Face recognition (ArcFace)
  face_db.hpp / face_db.cpp     Face database with FAISS/linear search
  face_demo.cpp                 Standalone demo binary
```

### Pipeline

```
Frame -> SCRFD (detect faces + 5-point landmarks)
           |
           v
       ArcFace (align face 112x112 -> extract 512-d embedding)
           |
           v
       FaceDatabase (L2-normalized cosine similarity search)
           |
           v
       Match result: (name, similarity) or "Unknown"
```

## Components

### SCRFD — Face Detection

ONNX-based face detector from InsightFace. Uses Feature Pyramid Network (FPN) with three stride levels for multi-scale detection.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | 640x640 | Model input resolution |
| `conf_threshold` | 0.5 | Minimum detection confidence |
| `nms_threshold` | 0.4 | NMS IoU threshold |
| `max_num` | 0 (all) | Max faces to return (0 = unlimited) |

**How it works:**
1. Letterbox resize input to model size (preserves aspect ratio, pads with black)
2. Normalize: `(pixel - 127.5) / 128.0`
3. Forward pass produces 9 outputs: 3 strides (8, 16, 32) x [scores, bboxes, keypoints]
4. Each stride has 2 anchors per grid cell
5. Decode bboxes as distance-to-edges from anchor center, keypoints as offsets
6. NMS to remove duplicates
7. If `max_num` set, keep largest faces by area

**Output:** `std::vector<FaceDetection>` where each detection contains:
- `cv::Rect2f bbox` — bounding box (x, y, width, height)
- `float score` — confidence score
- `std::array<cv::Point2f, 5> landmarks` — left eye, right eye, nose, left mouth, right mouth

**Available models:**

| Model | GFlops | Notes |
|-------|--------|-------|
| `det_10g.onnx` | ~10 | Best accuracy |
| `det_2.5g.onnx` | ~2.5 | Balanced |
| `det_500m.onnx` | ~0.5 | Fastest, mobile-friendly |

### ArcFace — Face Recognition

Extracts a 512-dimensional embedding vector from an aligned face image. Two faces of the same person will have high cosine similarity (>0.4 typically).

**How it works:**
1. Align face to canonical 112x112 pose using 5-point landmarks
   - Estimates similarity transform (`cv::estimateAffinePartial2D`) mapping detected landmarks to reference positions
   - Reference landmarks are hardcoded for the 112x112 template
2. Preprocess: RGB conversion, `(pixel - 127.5) / 127.5` normalization, CHW layout
3. ONNX inference produces 512-d raw embedding
4. Optional L2 normalization (enabled by default)

**Available models:**

| Model | Architecture | Notes |
|-------|-------------|-------|
| `w600k_mbf.onnx` | MobileFaceNet | Fast, good accuracy |
| `w600k_r50.onnx` | ResNet-50 | Best accuracy, slower |

Both trained on WebFace600K (600K identities).

### FaceDatabase — Similarity Search

Stores L2-normalized face embeddings with associated names. Supports two search backends:

**Linear scan (default fallback):**
- Brute-force cosine similarity against all entries
- O(n) per query, fine for small databases (<1000 faces)

**FAISS (optional, compile-time):**
- Uses `faiss::IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- Same accuracy as linear scan but optimized with SIMD
- Excels at batch search (multiple queries at once)
- Call `build_index()` after adding faces or loading from disk

**API:**

```cpp
FaceDatabase db;

// Add faces
db.add("alice", embedding_vector);
db.add("bob", embedding_vector);
db.build_index();  // rebuild FAISS index (no-op without FAISS)

// Search
auto [name, sim] = db.search(query_embedding, 0.4f);
// Returns ("alice", 0.72) or ("Unknown", 0.15)

// Batch search (FAISS-accelerated)
auto results = db.batch_search(embeddings, 0.4f);

// Management
db.remove("alice");     // remove by name
db.build_index();       // rebuild after remove
auto names = db.names();// list unique names
db.save("face_db.yaml");
db.load("face_db.yaml");

// Check backend
FaceDatabase::faiss_available(); // true if compiled with -DUSE_FAISS
```

**Persistence:** Uses OpenCV `FileStorage` (YAML format). Each entry stores name + embedding vector.

## Building

### With FAISS (recommended)

```bash
brew install faiss   # macOS
mkdir build && cd build
cmake ..             # auto-detects FAISS
make face_demo
```

### Without FAISS

```bash
cmake .. -DUSE_FAISS=OFF
make face_demo
```

The build output shows which backend is active:
```
-- FAISS found: /opt/homebrew/lib/libfaiss.dylib
```
or:
```
-- FAISS not found — falling back to linear search
```

## Usage

### Download models

```bash
./download_face_models.sh          # det_10g + w600k_mbf (default)
./download_face_models.sh --all    # all 5 models
```

### Run the demo

```bash
# Webcam with default models
./face_demo

# Webcam with specific models
./face_demo --det-model ../weights/det_10g.onnx --rec-model ../weights/w600k_mbf.onnx

# Video file
./face_demo --video input.mp4

# Build face DB from image directory (filename = person name)
./face_demo --faces-dir /path/to/faces/ --rebuild

# Custom threshold and confidence
./face_demo --threshold 0.5 --conf 0.3
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--webcam [N]` | 0 | Webcam index |
| `--video FILE` | — | Video file path |
| `--det-model PATH` | `../weights/det_10g.onnx` | SCRFD model |
| `--rec-model PATH` | `../weights/w600k_mbf.onnx` | ArcFace model |
| `--faces-dir PATH` | — | Directory of face images for batch registration |
| `--db PATH` | `face_db.yaml` | Database file path |
| `--threshold FLOAT` | 0.4 | Similarity threshold for recognition |
| `--conf FLOAT` | 0.5 | Detection confidence threshold |
| `--rebuild` | false | Force rebuild database from `--faces-dir` |

### Interactive Controls

| Key | Action |
|-----|--------|
| `r` | **Register** — freezes frame, numbers detected faces, press 1-9 to select, type name in terminal |
| `d` | **Delete** — lists registered names in terminal, type name to remove |
| `l` | **List** — print all registered names to terminal |
| `q` | **Quit** |

### Registration Modes

**Batch registration** (from image directory):
```bash
# Each image file = one person. Filename (without extension) = name.
# faces/
#   alice.jpg
#   bob.png
#   charlie.jpeg
./face_demo --faces-dir ./faces/ --rebuild
```

**Interactive registration** (from live feed):
1. Press `r` during live feed
2. Frame freezes, all detected faces are numbered
3. Press the number key (1-9) to select the face
4. Selected face is highlighted — type the person's name in the terminal
5. Embedding is extracted, saved to DB, and FAISS index rebuilt
6. Live feed resumes with the new face recognized

### Display

- **Recognized faces**: colored bounding box with `name: similarity` label (unique color per person)
- **Unknown faces**: red bounding box with "Unknown" label
- **No DB loaded**: green bounding boxes (detection only)
- **Landmarks**: yellow dots (5 points: eyes, nose, mouth corners)
- **HUD**: FPS, face count, DB size, control hints

## Thresholds

The similarity threshold (`--threshold`) controls the tradeoff:

| Threshold | Effect |
|-----------|--------|
| 0.3 | Loose — more matches, more false positives |
| 0.4 | **Default** — balanced |
| 0.5 | Strict — fewer false positives, might miss some |
| 0.6+ | Very strict — only very confident matches |

Cosine similarity ranges from -1 to 1 for L2-normalized embeddings. Same person typically scores 0.4-0.8+, different people typically score below 0.3.

## File Format

`face_db.yaml` (OpenCV FileStorage):
```yaml
%YAML:1.0
---
faces:
   - { name: "alice", embedding: [ 0.0123, -0.0456, ... ] }
   - { name: "bob", embedding: [ 0.0789, 0.0012, ... ] }
```

## Dependencies

| Library | Required | Purpose |
|---------|----------|---------|
| OpenCV | Yes | Image I/O, preprocessing, display, persistence |
| ONNX Runtime | Yes | Model inference (SCRFD, ArcFace) |
| FAISS | Optional | Fast similarity search (`-DUSE_FAISS=ON`) |
