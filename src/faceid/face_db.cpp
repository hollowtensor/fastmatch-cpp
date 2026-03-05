#include "face_db.hpp"
#include <cmath>
#include <algorithm>
#include <set>
#include <filesystem>
#include <iostream>

std::vector<float> FaceDatabase::normalize(const std::vector<float>& v) {
    float norm = 0;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    std::vector<float> out(v.size());
    if (norm > 1e-9f)
        for (size_t i = 0; i < v.size(); i++) out[i] = v[i] / norm;
    return out;
}

float FaceDatabase::cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0;
    for (size_t i = 0; i < a.size(); i++) dot += a[i] * b[i];
    return dot;  // assumes both are L2-normalized
}

bool FaceDatabase::faiss_available() {
#ifdef USE_FAISS
    return true;
#else
    return false;
#endif
}

bool FaceDatabase::remove(const std::string& name) {
    auto it = std::remove_if(faces_.begin(), faces_.end(),
        [&](const FaceRecord& r) { return r.name == name; });
    if (it == faces_.end()) return false;
    faces_.erase(it, faces_.end());
#ifdef USE_FAISS
    index_.reset();
#endif
    return true;
}

void FaceDatabase::add(const std::string& name, const std::vector<float>& embedding) {
    FaceRecord rec;
    rec.name = name;
    rec.embedding = normalize(embedding);
    faces_.push_back(std::move(rec));

#ifdef USE_FAISS
    // Invalidate index — needs rebuild
    index_.reset();
#endif
}

void FaceDatabase::build_index() {
#ifdef USE_FAISS
    if (faces_.empty()) return;
    dim_ = static_cast<int>(faces_[0].embedding.size());
    index_ = std::make_unique<faiss::IndexFlatIP>(dim_);

    // Add all embeddings in one batch
    std::vector<float> flat(faces_.size() * dim_);
    for (size_t i = 0; i < faces_.size(); i++)
        std::copy(faces_[i].embedding.begin(), faces_[i].embedding.end(),
                  flat.begin() + i * dim_);

    index_->add(static_cast<faiss::idx_t>(faces_.size()), flat.data());
#endif
}

std::pair<std::string, float> FaceDatabase::search_linear(
    const std::vector<float>& query, float threshold) const {
    float best_sim = -1.0f;
    int best_idx = -1;

    for (int i = 0; i < static_cast<int>(faces_.size()); i++) {
        float sim = cosine_sim(query, faces_[i].embedding);
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }

    if (best_sim > threshold && best_idx >= 0)
        return {faces_[best_idx].name, best_sim};
    return {"Unknown", best_sim};
}

std::pair<std::string, float> FaceDatabase::search(
    const std::vector<float>& embedding, float threshold) const {
    if (faces_.empty()) return {"Unknown", 0.0f};

    auto query = normalize(embedding);

#ifdef USE_FAISS
    if (index_ && index_->ntotal > 0) {
        float sim = 0;
        faiss::idx_t idx = -1;
        index_->search(1, query.data(), 1, &sim, &idx);

        if (idx >= 0 && sim > threshold)
            return {faces_[idx].name, sim};
        return {"Unknown", sim};
    }
#endif

    return search_linear(query, threshold);
}

std::vector<std::pair<std::string, float>> FaceDatabase::batch_search(
    const std::vector<std::vector<float>>& embeddings, float threshold) const {
    std::vector<std::pair<std::string, float>> results;
    results.reserve(embeddings.size());

#ifdef USE_FAISS
    if (index_ && index_->ntotal > 0 && !embeddings.empty()) {
        int n = static_cast<int>(embeddings.size());
        int d = static_cast<int>(embeddings[0].size());

        // Normalize and flatten
        std::vector<float> flat(n * d);
        for (int i = 0; i < n; i++) {
            auto normed = normalize(embeddings[i]);
            std::copy(normed.begin(), normed.end(), flat.begin() + i * d);
        }

        std::vector<float> sims(n);
        std::vector<faiss::idx_t> idxs(n);
        index_->search(n, flat.data(), 1, sims.data(), idxs.data());

        for (int i = 0; i < n; i++) {
            if (idxs[i] >= 0 && sims[i] > threshold)
                results.push_back({faces_[idxs[i]].name, sims[i]});
            else
                results.push_back({"Unknown", sims[i]});
        }
        return results;
    }
#endif

    for (const auto& emb : embeddings)
        results.push_back(search(emb, threshold));
    return results;
}

std::vector<std::string> FaceDatabase::names() const {
    std::set<std::string> unique;
    for (const auto& f : faces_) unique.insert(f.name);
    return {unique.begin(), unique.end()};
}

void FaceDatabase::save(const std::string& path) const {
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "faces" << "[";
    for (const auto& f : faces_) {
        fs << "{:" << "name" << f.name
           << "embedding" << std::vector<float>(f.embedding)
           << "}";
    }
    fs << "]";
    fs.release();
}

bool FaceDatabase::load(const std::string& path) {
    if (!std::filesystem::exists(path)) return false;

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    faces_.clear();
    cv::FileNode fn = fs["faces"];
    for (const auto& node : fn) {
        FaceRecord rec;
        rec.name = (std::string)node["name"];
        node["embedding"] >> rec.embedding;
        faces_.push_back(std::move(rec));
    }
    fs.release();

    build_index();

    std::cout << "Face DB: " << faces_.size() << " faces"
              << (faiss_available() ? " [FAISS]" : " [linear]") << std::endl;
    return true;
}
