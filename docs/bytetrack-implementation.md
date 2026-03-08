# ByteTrack Implementation

## Overview

ByteTrack is a multi-object tracking algorithm that associates **every detection box** — not just the high-confidence ones. It uses a Kalman filter for motion prediction and IoU-based Hungarian matching for data association, with a two-stage process that recovers occluded objects through low-confidence detections.

Our implementation is based on [Vertical-Beach/ByteTrack-cpp](https://github.com/Vertical-Beach/ByteTrack-cpp) (MIT license), adapted to work with the existing YOLO detector in fastmatch_cpp.

**Paper**: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)

## Architecture

```
src/bytetrack/
├── BYTETracker.h/.cpp   # Main tracker — orchestrates the 5-step update loop
├── STrack.h/.cpp        # Single track — Kalman state, lifecycle (New/Tracked/Lost/Removed)
├── KalmanFilter.h/.cpp  # 8-state Kalman filter (position + velocity in xyah space)
├── Rect.h/.cpp          # Bounding box with TLWH/TLBR/XYAH conversions and IoU
├── Object.h             # Detection input struct (rect + label + confidence)
└── lapjv.h/.cpp         # Jonker-Volgenant algorithm for optimal assignment
```

**Dependencies**: Eigen3 (matrix operations for Kalman filter), C++17.

## How It Works

### Input/Output

**Input**: A vector of `byte_track::Object` per frame — each is a bounding box (TLWH format) with a confidence score.

**Output**: A vector of `shared_ptr<STrack>` — active tracks with stable IDs, predicted bounding boxes, and confidence scores.

```cpp
byte_track::BYTETracker tracker(fps, track_buffer, track_thresh, high_thresh, match_thresh);

// Each frame:
std::vector<byte_track::Object> detections = /* from YOLO */;
auto tracks = tracker.update(detections);

for (auto& track : tracks) {
    track->getTrackId();   // stable ID
    track->getRect();      // Kalman-smoothed bounding box
    track->getScore();     // detection confidence
}
```

### The 5-Step Update Loop

Each call to `tracker.update()` runs these steps:

#### Step 1: Split Detections by Confidence

Detections are separated into two pools:

- **High-confidence**: `confidence >= track_thresh` (default 0.3)
- **Low-confidence**: `confidence < track_thresh`

Existing tracks are also split into **active** (currently tracked) and **non-active** (just created, not yet confirmed).

All active tracks and lost tracks are pooled together, and their Kalman filters are advanced one step via `predict()`.

#### Step 2: First Association (High-Confidence)

```
Existing tracks (active + lost)  ←→  High-confidence detections
```

An IoU cost matrix is computed between Kalman-predicted track positions and high-confidence detection boxes. The Jonker-Volgenant algorithm (LAPJV) finds the optimal assignment, discarding matches below `match_thresh` (default 0.8 cost = 0.2 IoU).

- **Matched**: Track state is updated with the detection. Lost tracks that match are **reactivated**.
- **Unmatched tracks**: Carried forward to Step 3.
- **Unmatched detections**: Carried forward to Step 4.

#### Step 3: Second Association (Low-Confidence)

```
Remaining unmatched tracks  ←→  Low-confidence detections
```

This is the key ByteTrack insight. Tracks that didn't match any high-confidence detection get a second chance against low-confidence detections, using a lower IoU threshold (0.5).

This recovers partially occluded objects that the detector reports with low confidence — they'd be discarded by other trackers but ByteTrack keeps them alive.

Tracks that still don't match after this step are marked as **Lost**.

#### Step 4: Initialize New Tracks

```
Non-active (unconfirmed) tracks  ←→  Remaining high-confidence detections
```

Unconfirmed tracks (created last frame but not yet validated) are matched against leftover high-confidence detections. If they match, they're promoted to active. If not, they're **Removed**.

Any remaining unmatched detections with confidence above `high_thresh` (default 0.4) become new tracks.

#### Step 5: State Management

- Lost tracks exceeding `max_time_lost` frames are permanently removed
- Duplicate tracks (IoU > 0.85) are deduplicated, keeping the longer-lived one
- Track lists are updated: `tracked_stracks_`, `lost_stracks_`, `removed_stracks_`

Only tracks that are both **Tracked** and **Activated** are returned.

### Kalman Filter

The Kalman filter operates in **XYAH** space (center-x, center-y, aspect-ratio, height) with an 8-dimensional state vector:

```
State: [cx, cy, a, h, v_cx, v_cy, v_a, v_h]
         position          velocity
```

- **Predict**: Applies constant-velocity motion model. Process noise scales with object height for size-adaptive tracking.
- **Update**: Standard Kalman update with the detection measurement. The bounding box is reconstructed from the updated state.
- Lost tracks have their velocity zeroed before prediction to prevent drift.

### Track Lifecycle

```
Detection → [New] → activate() → [Tracked] → markAsLost() → [Lost] → markAsRemoved() → [Removed]
                                      ↑                          |
                                      └── reActivate() ──────────┘
```

- **New**: Just created, not yet confirmed. Gets one frame to find a matching detection.
- **Tracked**: Actively matched to detections. Has a stable ID.
- **Lost**: No matching detection for recent frames. Still in the pool for potential re-association.
- **Removed**: Expired or duplicate. No longer considered.

### Linear Assignment (LAPJV)

The Jonker-Volgenant algorithm solves the linear assignment problem optimally — same result as the Hungarian algorithm but faster in practice. The cost matrix is extended to handle rectangular (non-square) cases by padding with dummy assignments.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_rate` | 30 | Video FPS. Scales `track_buffer` for time-consistent tracking. |
| `track_buffer` | 30 | Frames to keep lost tracks. At 30fps, this is 1 second. |
| `track_thresh` | 0.3 | Splits detections into high/low confidence pools. |
| `high_thresh` | 0.4 | Minimum confidence to initialize a new track. |
| `match_thresh` | 0.8 | IoU cost threshold (1 - IoU). 0.8 means minimum 0.2 IoU to match. |

The effective lost-track duration in seconds is: `track_buffer * (30 / frame_rate)`.

## Strengths and Limitations

**Strengths**:
- Frame-to-frame tracking is very fast (no feature extraction needed per frame)
- Two-stage association recovers partially occluded objects
- Kalman prediction handles brief occlusions and smooths trajectories
- Stable IDs while a person remains continuously visible

**Limitations**:
- Motion-only — no appearance model. Cannot re-identify someone after they leave and re-enter the frame.
- Linear motion assumption struggles with sudden direction changes.
- IoU-based matching fails when objects overlap heavily or move very fast between frames.
- Lost tracks expire after `track_buffer` frames with no way to recover identity.

These limitations are addressed by integrating Re-ID (see [reid-integration.md](reid-integration.md)).
