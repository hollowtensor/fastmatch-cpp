# Re-ID Integration with ByteTrack

## The Problem

ByteTrack assigns stable IDs while a person is continuously visible, but it has no memory of what someone **looks like**. When a person leaves the frame and returns — even one second later — ByteTrack assigns a completely new ID. In testing, a single person walking in and out of the webcam frame received 3 different IDs in under a minute.

This is a fundamental limitation of motion-only trackers: Kalman filters predict where someone **will be**, not who they **are**.

## The Solution

We add a Re-ID (re-identification) layer on top of ByteTrack that activates **only when ByteTrack creates a new track**. It uses OSNet appearance embeddings to check whether the "new" person is actually someone we've seen before.

```
ByteTrack handles:  Frame-to-frame continuity (Kalman + IoU)
Re-ID handles:      Identity recovery after disappearance
```

This is the same division of labor used by DeepSORT, BoT-SORT, and StrongSORT — but implemented as a lightweight post-processing layer rather than baked into the association step.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Per Frame                                              │
│                                                         │
│  YOLO Detector  ──→  ByteTrack Tracker  ──→  Tracks     │
│                                               │         │
│                          ┌────────────────────┘         │
│                          ▼                              │
│                   For each track:                       │
│                   ┌─────────────────────┐               │
│                   │ New ByteTrack ID?   │               │
│                   │                     │               │
│                   │  YES ──→ Extract    │               │
│                   │          embedding  │               │
│                   │          ▼          │               │
│                   │   Match against     │               │
│                   │   lost gallery      │               │
│                   │          ▼          │               │
│                   │   dist < thresh?    │               │
│                   │   YES: recover ID   │               │
│                   │   NO:  assign new   │               │
│                   │                     │               │
│                   │  NO ──→ Update      │               │
│                   │         embedding   │               │
│                   └─────────────────────┘               │
│                                                         │
│  When a track is lost:                                  │
│    Save its embedding to the lost gallery               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Key Data Structures

### Display ID Mapping

ByteTrack's internal IDs are ephemeral — a new one is minted each time a track is created. We maintain a mapping from ByteTrack's internal IDs to persistent **display IDs** that survive re-entries:

```cpp
std::unordered_map<size_t, size_t> bt_to_display;
// bt_to_display[bytetrack_id] = display_id
```

### Active Embeddings

For each currently-tracked person, we store their latest Re-ID embedding:

```cpp
std::unordered_map<size_t, std::vector<float>> active_embeddings;
// active_embeddings[bytetrack_id] = 512-d feature vector
```

These are extracted every frame from the person's bounding box crop using the OSNet model.

### Lost Gallery

When a track disappears, its embedding is saved to the lost gallery:

```cpp
struct LostTrackEntry {
    size_t display_id;            // the persistent ID
    std::vector<float> embedding; // appearance at time of loss
    int lost_frame;               // for TTL expiry
};

std::deque<LostTrackEntry> lost_gallery;
```

Entries expire after `lost_ttl` seconds (default 30s) to prevent stale matches.

## How Recovery Works

### 1. Detecting Lost Tracks

Each frame, we compare the set of active ByteTrack IDs against the previous frame. Any ID that was active last frame but is absent now has been lost:

```cpp
for (auto& [bt_id, _] : prev_active) {
    if (!curr_active.count(bt_id) && active_embeddings.count(bt_id)) {
        // This track just disappeared — save its embedding
        lost_gallery.push_back({display_id, embedding, frame_num});
    }
}
```

### 2. Matching New Tracks

When ByteTrack creates a new track (a ByteTrack ID we haven't seen before), we extract its embedding and compare against every entry in the lost gallery using cosine distance:

```cpp
float best_dist = 1e9f;
for (auto& entry : lost_gallery) {
    float d = cosine_distance(new_embedding, entry.embedding);
    if (d < best_dist) best_dist = d;
}

if (best_dist < reid_thresh) {
    // This "new" person is actually someone we've seen before
    bt_to_display[new_bt_id] = recovered_display_id;
}
```

### 3. What Cosine Distance Means

The OSNet model produces a 512-dimensional embedding for each person crop. Cosine distance measures how different two embeddings are:

- **0.0**: Identical (same person, same pose)
- **~0.15-0.25**: Same person, different pose/angle
- **~0.35-0.45**: Boundary — might be same person or different
- **>0.50**: Almost certainly different people

The default threshold of **0.40** was chosen empirically. In testing, re-entries matched at distances of 0.17 and 0.27 — well within the threshold.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--reid` | off | Enable Re-ID recovery |
| `--reid-thresh` | 0.40 | Cosine distance threshold. Lower = stricter matching, fewer false recoveries. |
| `--lost-ttl` | 30 | Seconds to keep lost track embeddings. Longer = more chance of recovery but higher risk of confusion. |

The threshold from `config.yaml` (`feature_extraction_threshold`) is used as the default when `--reid-thresh` is not specified.

## Performance Impact

Without Re-ID, the demo runs at pure ByteTrack speed — only detection + Kalman/IoU.

With Re-ID enabled, there is additional cost per tracked person per frame:
- **One OSNet inference per tracked person** (~2-5ms per crop on CPU)
- **One linear scan of lost gallery** per new track (negligible)

For a typical scene with 1-5 people, this adds 5-25ms per frame. The Re-ID model is the same OSNet (osnet_ain_x1_0) used by the main fastmatch pipeline.

## Why Not Embed Re-ID Into the Association Step?

Some trackers (DeepSORT, StrongSORT) use appearance features **during** the association step — combining IoU and Re-ID distance into a joint cost matrix. We chose the simpler post-processing approach for several reasons:

1. **ByteTrack's IoU matching is already very good** for frame-to-frame tracking. Adding Re-ID to every association step would slow it down without improving continuous tracking.

2. **Re-ID is expensive**. Running OSNet on every detection every frame for association would multiply inference cost. By only running it on new tracks, we minimize the overhead.

3. **Clean separation of concerns**. ByteTrack handles motion, Re-ID handles identity. Each can be tuned independently. The `--reid` flag makes it easy to A/B test.

4. **Same approach scales to multi-camera**. The lost gallery concept extends naturally: tracks lost on camera A can be recovered on camera B using the same cosine matching.

## Observed Results

In webcam testing (single person, walking in/out of frame):

| Mode | IDs Assigned | Correct |
|------|-------------|---------|
| ByteTrack only | 3 | No — new ID each re-entry |
| ByteTrack + Re-ID | 1 | Yes — recovered twice at dist=0.17 and dist=0.27 |
