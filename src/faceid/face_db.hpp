#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#ifdef USE_FAISS
#include <faiss/IndexFlat.h>
#endif

struct FaceRecord {
    std::string name;
    std::vector<float> embedding;  // L2-normalized
};

class FaceDatabase {
public:
    FaceDatabase() = default;

    void add(const std::string& name, const std::vector<float>& embedding);

    // Remove all entries with given name. Returns true if any removed.
    bool remove(const std::string& name);

    // Search for best match. Returns (name, similarity). "Unknown" if below threshold.
    std::pair<std::string, float> search(const std::vector<float>& embedding,
                                          float threshold = 0.4f) const;

    // Batch search — FAISS excels here
    std::vector<std::pair<std::string, float>> batch_search(
        const std::vector<std::vector<float>>& embeddings,
        float threshold = 0.4f) const;

    int size() const { return static_cast<int>(faces_.size()); }

    void save(const std::string& path) const;
    bool load(const std::string& path);

    std::vector<std::string> names() const;

    // Rebuild FAISS index after load or bulk add
    void build_index();

    static bool faiss_available();

private:
    static float cosine_sim(const std::vector<float>& a, const std::vector<float>& b);
    static std::vector<float> normalize(const std::vector<float>& v);

    // Linear scan fallback
    std::pair<std::string, float> search_linear(const std::vector<float>& query,
                                                 float threshold) const;

    std::vector<FaceRecord> faces_;

#ifdef USE_FAISS
    mutable std::unique_ptr<faiss::IndexFlatIP> index_;
    int dim_ = 0;
#endif
};
