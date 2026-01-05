# C++ DPVO Implementation for Ambarella CV28
## Simplified 3-Thread Pipeline Architecture

## Overview

This outline describes a simplified 3-thread pipeline architecture for DPVO on Ambarella CV28. The design uses **eazyai library** for neural network inference and a simple producer-consumer pipeline model.

### Key Constraints for CV28:
- ❌ **No PyTorch**: CV28 does not support PyTorch
- ❌ **No CUDA**: CV28 has no CUDA support - all operations must be CPU-based
- ✅ **EazyAI**: Use Ambarella's eazyai library for neural network inference
- ✅ **CVflow Processor**: Neural networks run on dedicated CVflow hardware (via eazyai)
- ✅ **CPU Operations**: Correlation, BA, and projective operations run on CPU (with SIMD optimization)

---

## 1. Architecture: 3-Thread Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Thread 1: Image Acquisition              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │ Sensor/ │ ───> │ Preproc  │ ───> │  Queue   │          │
│  │ Video   │      │ (resize)  │      │  (raw)   │          │
│  └──────────┘      └──────────┘      └──────────┘          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Thread 2: Patchify (Feature Extraction)       │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  Queue   │ ───> │ 1. fnet  │ ───> │ 2. Patchify│        │
│  │  (raw)   │      │ inference│      │ fnet/inet │        │
│  └──────────┘      │ 2. inet  │      │ /image    │        │
│                    │ inference│      └──────────┘         │
│                    │ (EazyAI) │                            │
│                    └──────────┘                            │
│  Output: fmap, imap, gmap, patches, colors                │
│  └──────────────────────────────────────────────────────┘  │
│                    │                                        │
│                    ▼                                        │
│            ┌──────────────┐                                 │
│            │ Queue (patch │                                 │
│            │  data ready) │                                 │
│            └──────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Thread 3: Update (Optimization)               │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  Queue   │ ───> │ 1. Corr  │ ───> │ 2. Update│         │
│  │(patch    │      │          │      │ Model    │         │
│  │ data)    │      │          │      │ Inference│         │
│  └──────────┘      └──────────┘      └──────────┘         │
│                            │                                │
│                            ▼                                │
│                    ┌──────────────┐                        │
│                    │ 3. BA       │ ───> │  Poses  │        │
│                    │ Function    │      │ Points  │        │
│                    └──────────────┘      └─────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Thread Responsibilities

**Thread 1: Image Acquisition**
- Read frames from sensor/video
- Preprocess (resize, crop, normalize if needed)
- Enqueue raw images with timestamps
- Non-blocking, runs continuously

**Thread 2: Patchify (Feature Extraction)**
1. **fnet & inet inference** (EazyAI on CVflow):
   - Dequeue raw image from Thread 1
   - Run fnet inference → get fmap (feature map)
   - Run inet inference → get imap (instance feature map)
   
2. **Patchify operations** (CPU-based):
   - Patchify fmap → extract gmap (global patch features)
   - Patchify inet → extract imap (local patch features)  
   - Patchify image → extract patches (coordinates + depth) and colors
   - Select M patches per frame (random or gradient-based)
   
3. **Enqueue processed data**:
   - Put fmap, imap, gmap, patches, colors into queue for Thread 3
   - Include metadata: timestamp, intrinsics, sequence number

**Thread 3: Update (Optimization)**
1. **Correlation** (CPU-based, SIMD optimized):
   - Dequeue patchified data from Thread 2
   - Compute correlation volume between patches and frames
   - Reproject patches to target frames
   
2. **Update model inference** (EazyAI on CVflow):
   - Run update network inference
   - Input: net, context (imap), correlation
   - Output: delta (flow correction), weight (confidence)
   
3. **BA function** (CPU-based, Eigen solver):
   - Bundle adjustment optimization
   - Update poses and patch depths
   - Update PatchGraph state
   - Generate point cloud

---

## 2. Will This Work? Analysis

### ✅ Advantages

1. **Simple and Clear**: Easy to understand and debug
2. **Pipeline Parallelism**: While Thread 3 processes frame N, Thread 2 processes frame N+1, Thread 1 captures frame N+2
3. **Natural Synchronization**: Queues provide automatic synchronization
4. **Resource Isolation**: Each thread has clear responsibilities
5. **Suitable for Embedded**: Low overhead, predictable behavior

### ⚠️ Potential Issues & Solutions

#### Issue 1: Frame Ordering
**Problem**: If Thread 2 or 3 is slow, frames may arrive out of order.

**Solution**: Use priority queue or maintain frame sequence numbers
```cpp
struct FrameData {
    int64_t timestamp;
    int sequence_number;
    cv::Mat image;
    Eigen::Vector4f intrinsics;
};
```

#### Issue 2: Queue Overflow
**Problem**: If Thread 3 is much slower, queue between Thread 2-3 may overflow.

**Solution**: 
- Limit queue size (e.g., max 3-5 frames)
- Drop oldest frames if queue is full
- Or: Block Thread 2 until queue has space (backpressure)

#### Issue 3: State Synchronization
**Problem**: Thread 3 needs exclusive access to PatchGraph state.

**Solution**: Thread 3 has exclusive ownership of PatchGraph. No other thread accesses it.

#### Issue 4: Initialization
**Problem**: Need to accumulate frames before starting optimization.

**Solution**: Thread 3 waits until enough frames accumulated (e.g., n >= 8).

---

## 3. Detailed Thread Design

### 3.1 Thread 1: Image Acquisition

```cpp
class ImageAcquisitionThread {
public:
    ImageAcquisitionThread(
        std::shared_ptr<ThreadSafeQueue<RawFrame>> output_queue,
        const Config& cfg);
    
    void run();
    void stop();
    
private:
    std::shared_ptr<ThreadSafeQueue<RawFrame>> output_queue_;
    Config cfg_;
    std::atomic<bool> running_{true};
    
    // Image source (sensor or video)
    std::unique_ptr<ImageSource> source_;
    
    void process_loop();
};

struct RawFrame {
    int64_t timestamp;
    int sequence_number;
    cv::Mat image;              // Raw image from sensor/video
    Eigen::Vector4f intrinsics;  // Camera intrinsics [fx, fy, cx, cy]
};
```

**Implementation Notes:**
- Non-blocking image capture
- Minimal preprocessing (just resize/crop if needed)
- Timestamp each frame
- Drop frames if output queue is full (or use blocking queue)

### 3.2 Thread 2: Patchify (Feature Extraction)

```cpp
class PatchifyThread {
public:
    PatchifyThread(
        std::shared_ptr<ThreadSafeQueue<RawFrame>> input_queue,
        std::shared_ptr<ThreadSafeQueue<FeatureFrame>> output_queue,
        const Config& cfg,
        const std::string& fnet_model_path,  // CV28 model path (.bin/.amb)
        const std::string& inet_model_path); // CV28 model path (.bin/.amb)
    
    void run();
    void stop();
    
private:
    std::shared_ptr<ThreadSafeQueue<RawFrame>> input_queue_;
    std::shared_ptr<ThreadSafeQueue<FeatureFrame>> output_queue_;
    Config cfg_;
    std::atomic<bool> running_{true};
    
    // EazyAI inference engines (CV28)
    std::unique_ptr<EazyAIInference> fnet_;  // Feature network
    std::unique_ptr<EazyAIInference> inet_;   // Instance network
    
    void process_loop();
    FeatureFrame patchify_frame(const RawFrame& raw_frame);
    
    // Step 1: Run fnet & inet inference
    std::pair<Eigen::Tensor<float, 4>, Eigen::Tensor<float, 4>> 
        run_inference(const cv::Mat& image);
    
    // Step 2: Patchify fnet/inet/image
    FeatureFrame extract_patches(
        const cv::Mat& image,
        const Eigen::Tensor<float, 4>& fmap,
        const Eigen::Tensor<float, 4>& imap);
};

struct FeatureFrame {
    int64_t timestamp;
    int sequence_number;
    
    // Feature maps (from patchify)
    Eigen::Tensor<float, 4> fmap;      // [1, 128, H/4, W/4] - Full feature map (for correlation)
    Eigen::Tensor<float, 3> imap;      // [M, DIM, 1, 1] - Local patch features (from inet)
    Eigen::Tensor<float, 5> gmap;      // [M, 128, P, P] - Global patch features (from fnet)
    
    // Patches (from patchify)
    Eigen::Tensor<float, 5> patches;   // [M, 3, P, P] - Patch coordinates + depth
    Eigen::Matrix<uint8_t, 3> colors;  // [M, 3] - Patch colors (from image)
    
    // Metadata
    Eigen::Vector4f intrinsics;
    std::vector<int> patch_indices;    // Which frame each patch belongs to
};
```

**Implementation Notes:**
- **Step 1**: Run fnet & inet inference using EazyAI (CVflow processor)
- **Step 2**: Patchify fmap/inet/image to extract patches (CPU-based)
- **Step 3**: Enqueue all processed data for Thread 3
- Process one frame at a time
- Extract M patches per frame (random or gradient-based selection)
- Output queue size limited (e.g., max 3 frames) to prevent memory buildup

### 3.3 Thread 3: Update (Optimization)

```cpp
class UpdateThread {
public:
    UpdateThread(
        std::shared_ptr<ThreadSafeQueue<FeatureFrame>> input_queue,
        const Config& cfg,
        const std::string& update_model_path);  // CV28 model path (.bin/.amb)
    
    void run();
    void stop();
    
    // State queries (thread-safe)
    Eigen::MatrixXf get_poses() const;
    Eigen::MatrixXf get_points() const;
    bool is_initialized() const;
    
private:
    std::shared_ptr<ThreadSafeQueue<FeatureFrame>> input_queue_;
    Config cfg_;
    std::atomic<bool> running_{true};
    
    // EazyAI for update network (CV28)
    std::unique_ptr<EazyAIInference> update_net_;
    
    // DPVO state (exclusive to this thread)
    std::unique_ptr<DPVOState> state_;
    std::unique_ptr<PatchGraph> pg_;
    
    // Synchronization for state queries
    mutable std::shared_mutex state_mutex_;
    
    void process_loop();
    void process_frame(const FeatureFrame& features);
    
    // Step 1: Correlation
    Eigen::Tensor<float, 3> compute_correlation(
        const Eigen::Tensor<float, 5>& coords);
    
    // Step 2: Update model inference
    std::tuple<Eigen::Tensor<float, 3>, Eigen::MatrixXf, Eigen::MatrixXf>
        run_update_inference(
            const Eigen::Tensor<float, 3>& net,
            const Eigen::Tensor<float, 3>& ctx,
            const Eigen::Tensor<float, 3>& corr,
            const std::vector<int>& ii,
            const std::vector<int>& jj,
            const std::vector<int>& kk);
    
    // Step 3: BA function
    void run_bundle_adjustment(
        const Eigen::MatrixXf& target,
        const Eigen::MatrixXf& weight,
        const std::vector<int>& ii,
        const std::vector<int>& jj,
        const std::vector<int>& kk);
    
    void update();
    void keyframe();
};
```

**Implementation Notes:**
- **Step 1**: Compute correlation (CPU-based, SIMD optimized)
  - Reproject patches to target frames
  - Compute correlation volume between patches and frames
  
- **Step 2**: Update model inference (EazyAI on CVflow)
  - Input: net state, context (imap), correlation
  - Output: delta (flow correction), weight (confidence)
  
- **Step 3**: BA function (CPU-based, Eigen solver)
  - Bundle adjustment optimization
  - Update poses and patch depths
  - Update PatchGraph state
  
- Owns all DPVO state (PatchGraph, poses, patches, etc.)
- Processes frames sequentially
- Most computationally intensive thread

---

## 4. Thread-Safe Queue Implementation

```cpp
template<typename T>
class ThreadSafeQueue {
public:
    void push(const T& item);
    void push(T&& item);
    
    bool try_pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(0));
    void pop(T& item);  // Blocking
    
    bool empty() const;
    size_t size() const;
    
    // Limit queue size
    void set_max_size(size_t max_size);
    
private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<T> queue_;
    size_t max_size_ = 10;  // Default max size
};
```

**Queue Sizing Strategy:**
- Thread 1 → Thread 2: Large queue (10-20 frames) - sensor may be fast
- Thread 2 → Thread 3: Small queue (3-5 frames) - prevent memory buildup, provide backpressure

---

## 5. Main DPVO Class (Orchestrator)

```cpp
class DPVO {
public:
    DPVO(const Config& cfg,
         const std::string& fnet_model_path,  // CV28 model paths
         const std::string& inet_model_path,
         const std::string& update_model_path,
         int ht = 480, int wd = 640);
    
    ~DPVO();
    
    // Start the pipeline
    void start();
    
    // Stop the pipeline
    void stop();
    
    // State queries (thread-safe)
    Eigen::MatrixXf get_poses() const;
    Eigen::MatrixXf get_points() const;
    bool is_initialized() const;
    
    // Get final trajectory (call after stop())
    std::pair<Eigen::MatrixXf, std::vector<double>> get_trajectory();
    
private:
    Config cfg_;
    
    // Thread-safe queues
    std::shared_ptr<ThreadSafeQueue<RawFrame>> raw_queue_;
    std::shared_ptr<ThreadSafeQueue<FeatureFrame>> feature_queue_;
    
    // Threads
    std::unique_ptr<ImageAcquisitionThread> acquisition_thread_;
    std::unique_ptr<PatchifyThread> patchify_thread_;
    std::unique_ptr<UpdateThread> update_thread_;
    
    // Thread handles
    std::thread acquisition_thread_handle_;
    std::thread patchify_thread_handle_;
    std::thread update_thread_handle_;
};
```

---

## 6. EazyAI Integration (Ambarella CV28)

### 6.1 EazyAI Inference Wrapper

```cpp
#include <eazyai/eazyai.h>  // Ambarella eazyai library

class EazyAIInference {
public:
    EazyAIInference(const std::string& model_path,
                   const std::string& model_name);
    ~EazyAIInference();
    
    // Initialize model (load and prepare for inference)
    bool initialize();
    
    // Run inference
    // Note: Exact API depends on eazyai version - adjust accordingly
    bool run_inference(const float* input_data, 
                      size_t input_size,
                      float* output_data,
                      size_t output_size);
    
    // Convenience methods for DPVO
    Eigen::Tensor<float, 4> run_fnet(const cv::Mat& image);
    Eigen::Tensor<float, 4> run_inet(const cv::Mat& image);
    std::tuple<Eigen::Tensor<float, 3>, Eigen::MatrixXf, Eigen::MatrixXf>
        run_update(const Eigen::Tensor<float, 3>& net,
                   const Eigen::Tensor<float, 3>& ctx,
                   const Eigen::Tensor<float, 3>& corr,
                   const std::vector<int>& ii,
                   const std::vector<int>& jj,
                   const std::vector<int>& kk);
    
    // Get input/output shapes
    std::vector<int> get_input_shape() const;
    std::vector<int> get_output_shape() const;
    
private:
    std::string model_path_;
    std::string model_name_;
    
    // EazyAI model handle (adjust type based on actual eazyai API)
    void* model_handle_;  // or specific eazyai type
    
    // Input/output buffers (pre-allocated for performance)
    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;
    
    // Shape information
    std::vector<int> input_shape_;
    std::vector<int> output_shape_;
    
    // Helper methods
    cv::Mat preprocess_image(const cv::Mat& image);
    void convert_to_eigen(const float* data, Eigen::Tensor<float, 4>& tensor, 
                        const std::vector<int>& shape);
    void convert_from_eigen(const Eigen::Tensor<float, 3>& tensor, 
                           float* data);
};
```

### 6.2 EazyAI Implementation Example

```cpp
// Example implementation (adjust based on actual eazyai API)
EazyAIInference::EazyAIInference(const std::string& model_path,
                                 const std::string& model_name)
    : model_path_(model_path), model_name_(model_name), model_handle_(nullptr) {
}

bool EazyAIInference::initialize() {
    // Load model using eazyai API
    // Example (adjust based on actual API):
    // model_handle_ = eazyai_load_model(model_path_.c_str());
    // if (!model_handle_) return false;
    
    // Get input/output shapes
    // input_shape_ = eazyai_get_input_shape(model_handle_);
    // output_shape_ = eazyai_get_output_shape(model_handle_);
    
    // Pre-allocate buffers
    size_t input_size = 1;
    for (int dim : input_shape_) input_size *= dim;
    input_buffer_.resize(input_size);
    
    size_t output_size = 1;
    for (int dim : output_shape_) output_size *= dim;
    output_buffer_.resize(output_size);
    
    return true;
}

Eigen::Tensor<float, 4> EazyAIInference::run_fnet(const cv::Mat& image) {
    // 1. Preprocess image
    cv::Mat processed = preprocess_image(image);
    
    // 2. Convert to input buffer format
    // (eazyai typically expects NHWC or NCHW format)
    // Adjust based on model requirements
    size_t idx = 0;
    for (int h = 0; h < processed.rows; ++h) {
        for (int w = 0; w < processed.cols; ++w) {
            cv::Vec3b pixel = processed.at<cv::Vec3b>(h, w);
            input_buffer_[idx++] = pixel[0] / 255.0f;
            input_buffer_[idx++] = pixel[1] / 255.0f;
            input_buffer_[idx++] = pixel[2] / 255.0f;
        }
    }
    
    // 3. Run inference
    // eazyai_run_inference(model_handle_, input_buffer_.data(), 
    //                     output_buffer_.data());
    
    // 4. Convert output to Eigen tensor
    Eigen::Tensor<float, 4> result;
    convert_to_eigen(output_buffer_.data(), result, output_shape_);
    
    return result;
}
```

### 6.3 Model Preparation for CV28

Since Ambarella CV28 uses eazyai, you need to:

1. **Convert models to CV28 format**:
   - Export PyTorch models to ONNX (already done: `fnet.onnx`, `inet.onnx`, `update.onnx`)
   - Use Ambarella's model conversion tools to convert ONNX to CV28 format
   - CV28 models typically have `.bin` or `.amb` extension

2. **Model optimization for CV28**:
   - Use Ambarella's model optimization tools
   - Quantize models if needed (INT8 quantization)
   - Optimize for CV28's CVflow architecture

3. **Model deployment**:
   - Place converted models on CV28 filesystem
   - Load models using eazyai API at initialization
   - Models run on CV28's dedicated CVflow processor (not CPU)

### 6.4 CV28 Model Conversion Workflow

```
PyTorch Model → ONNX → Ambarella Converter → CV28 Model (.bin/.amb)
     ↓              ↓            ↓                    ↓
  fnet.pth    fnet.onnx    CV28 Tools         fnet.bin
  inet.pth    inet.onnx    CV28 Tools         inet.bin
  update.pth  update.onnx  CV28 Tools         update.bin
```

**Note**: Consult Ambarella's documentation for:
- Exact eazyai API calls
- Model conversion tools and commands
- CV28-specific optimizations
- Memory management requirements

---

## 7. Data Structures

### 7.1 PatchGraph (Simplified for Single-Thread Access)

```cpp
class PatchGraph {
public:
    PatchGraph(const Config& cfg, int P, int DIM, int pmem);
    
    // Frame data
    int n = 0;  // number of frames
    int m = 0;  // number of patches
    
    // Static tensors (pre-allocated)
    Eigen::MatrixXf poses_;           // [N, 7]
    Eigen::Tensor<float, 5> patches_; // [N, M, 3, P, P]
    Eigen::MatrixXf intrinsics_;     // [N, 4]
    Eigen::MatrixXf points_;          // [N*M, 3]
    Eigen::Matrix<uint8_t, 3> colors_; // [N, M, 3]
    std::vector<int64_t> tstamps_;
    
    // Edge data
    int num_edges = 0;
    int max_edges;
    std::vector<int> ii, jj, kk;
    Eigen::Tensor<float, 3> net;
    Eigen::MatrixXf target, weight;
    
    // Operations (called only from UpdateThread)
    void append_factors(const std::vector<int>& ii, 
                       const std::vector<int>& jj);
    void remove_factors(const std::vector<bool>& mask, bool store);
    void normalize();
    
private:
    Config cfg_;
    int P, DIM, pmem, M, N;
};
```

**Note**: Since only UpdateThread accesses PatchGraph, no mutex needed!

### 7.2 DPVO State

```cpp
class DPVOState {
public:
    DPVOState(const Config& cfg, int DIM, int RES, int P, int M, int N, int pmem, int mem);
    
    // Feature maps (circular buffers)
    Eigen::Tensor<float, 3> imap_;  // [pmem, M, DIM]
    Eigen::Tensor<float, 5> gmap_;  // [pmem, M, 128, P, P]
    Eigen::Tensor<float, 5> fmap1_; // [1, mem, 128, H, W]
    Eigen::Tensor<float, 5> fmap2_; // [1, mem, 128, H/4, W/4]
    
    // State flags
    bool is_initialized = false;
    int counter = 0;
    std::vector<bool> ran_global_ba;
    int last_global_ba = -1000;
    
    // Timestamps
    std::vector<double> tlist;
    
private:
    Config cfg_;
    int DIM, RES, P, M, N, pmem, mem;
};
```

---

## 8. Implementation Flow

### 8.1 Thread 2: Patchify Implementation Flow

```cpp
void PatchifyThread::process_loop() {
    while (running_) {
        RawFrame raw_frame;
        if (!input_queue_->try_pop(raw_frame, std::chrono::milliseconds(100))) {
            continue;  // No frame available
        }
        
        // Process frame through patchify pipeline
        FeatureFrame feature_frame = patchify_frame(raw_frame);
        
        // Enqueue for Thread 3
        output_queue_->push(feature_frame);
    }
}

FeatureFrame PatchifyThread::patchify_frame(const RawFrame& raw_frame) {
    FeatureFrame result;
    result.timestamp = raw_frame.timestamp;
    result.sequence_number = raw_frame.sequence_number;
    result.intrinsics = raw_frame.intrinsics;
    
    // Step 1: Run fnet & inet inference (EazyAI on CVflow)
    auto [fmap, imap_full] = run_inference(raw_frame.image);
    
    // Step 2: Patchify fnet/inet/image
    result = extract_patches(raw_frame.image, fmap, imap_full);
    
    return result;
}

std::pair<Eigen::Tensor<float, 4>, Eigen::Tensor<float, 4>> 
PatchifyThread::run_inference(const cv::Mat& image) {
    // Preprocess image: normalize to [-0.5, 0.5]
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 2.0/255.0, -0.5);
    
    // Run fnet inference (EazyAI on CVflow)
    Eigen::Tensor<float, 4> fmap = fnet_->run_fnet(normalized);
    // fmap shape: [1, 128, H/4, W/4]
    
    // Run inet inference (EazyAI on CVflow)
    Eigen::Tensor<float, 4> imap_full = inet_->run_inet(normalized);
    // imap_full shape: [1, DIM, H/4, W/4]
    
    return {fmap, imap_full};
}

FeatureFrame PatchifyThread::extract_patches(
    const cv::Mat& image,
    const Eigen::Tensor<float, 4>& fmap,
    const Eigen::Tensor<float, 4>& imap_full) {
    
    FeatureFrame result;
    
    int M = cfg_.PATCHES_PER_FRAME;
    int P = 3;  // Patch size
    int H = fmap.dimension(2);  // H/4
    int W = fmap.dimension(3);  // W/4
    
    // Select patch centers (random or gradient-based)
    std::vector<std::pair<int, int>> patch_centers;
    if (cfg_.CENTROID_SEL_STRAT == "RANDOM") {
        for (int i = 0; i < M; ++i) {
            int x = rand() % (W - 2) + 1;  // Avoid edges
            int y = rand() % (H - 2) + 1;
            patch_centers.push_back({x, y});
        }
    } else if (cfg_.CENTROID_SEL_STRAT == "GRADIENT_BIAS") {
        // Select patches with high gradient (more complex, see original code)
        // ... gradient-based selection ...
    }
    
    // Patchify fmap → gmap (global patch features)
    result.gmap = patchify_tensor(fmap, patch_centers, P/2);
    // gmap shape: [M, 128, P, P]
    
    // Patchify imap_full → imap (local patch features)
    result.imap = patchify_tensor(imap_full, patch_centers, 0);
    // imap shape: [M, DIM, 1, 1]
    
    // Patchify image → patches (coordinates + depth) and colors
    result.patches = patchify_image_coords(image, patch_centers, P/2);
    // patches shape: [M, 3, P, P] - (x, y, depth)
    
    result.colors = patchify_image_colors(image, patch_centers, 0);
    // colors shape: [M, 3] - RGB
    
    // Store full fmap for correlation (Thread 3 needs it)
    result.fmap = fmap;  // [1, 128, H/4, W/4]
    
    return result;
}
```

### 8.2 Thread 3: Update Implementation Flow

### 8.3 Frame Processing in UpdateThread

```cpp
void UpdateThread::process_frame(const FeatureFrame& features) {
    // 1. Update PatchGraph with new frame
    {
        std::lock_guard<std::shared_mutex> lock(state_mutex_);
        
        // Add frame to PatchGraph
        int frame_idx = pg_->n;
        pg_->tstamps_[frame_idx] = features.timestamp;
        pg_->intrinsics_.row(frame_idx) = features.intrinsics;
        pg_->patches_.slice(...) = features.patches;
        pg_->colors_.slice(...) = features.colors;
        
        // Update feature maps (circular buffer)
        int idx = frame_idx % state_->pmem;
        state_->imap_.slice(...) = features.imap;
        state_->gmap_.slice(...) = features.gmap;
        
        idx = frame_idx % state_->mem;
        state_->fmap1_.slice(...) = features.fmap;
        state_->fmap2_.slice(...) = downsample(features.fmap, 4);
        
        // Initialize pose (motion model)
        if (frame_idx > 1) {
            // Damped linear motion model
            Eigen::VectorXf pose = predict_pose(pg_->poses_.row(frame_idx-1),
                                                 pg_->poses_.row(frame_idx-2),
                                                 state_->tlist);
            pg_->poses_.row(frame_idx) = pose;
        }
        
        pg_->n++;
        pg_->m += cfg_.PATCHES_PER_FRAME;
        state_->counter++;
        state_->tlist.push_back(features.timestamp);
    }
    
    // 2. Add edges (forward and backward)
    if (pg_->n > 1) {
        auto [ii_forw, jj_forw] = edges_forward();
        auto [ii_back, jj_back] = edges_backward();
        pg_->append_factors(ii_forw, jj_forw);
        pg_->append_factors(ii_back, jj_back);
    }
    
    // 3. Initialize if needed
    if (pg_->n == 8 && !state_->is_initialized) {
        state_->is_initialized = true;
        for (int i = 0; i < 12; ++i) {
            update();
        }
    }
    
    // 4. Update and keyframe
    if (state_->is_initialized) {
        update();
        keyframe();
    }
}
```

### 8.4 Update Function (Detailed Steps)

```cpp
void UpdateThread::update() {
    // Step 1: Correlation (CPU-based, SIMD optimized)
    // 1.1 Reproject patches to target frames
    auto coords = reproject();
    // coords shape: [1, num_edges, 2, P, P] - reprojected coordinates
    
    // 1.2 Compute correlation volume
    auto corr = compute_correlation(coords);
    // corr shape: [1, num_edges, correlation_dim] - correlation features
    
    // Step 2: Update model inference (EazyAI on CVflow)
    // 2.1 Prepare inputs
    auto ctx = state_->imap_;  // Context from imap
    // ctx shape: [1, num_edges, DIM] - patch context features
    
    // 2.2 Run update network inference
    auto [net_new, delta, weight] = run_update_inference(
        pg_->net,      // Current net state
        ctx,           // Context (imap)
        corr,           // Correlation
        pg_->ii,       // Source frame indices
        pg_->jj,       // Target frame indices
        pg_->kk);      // Patch indices
    
    // 2.3 Update net state
    pg_->net = net_new;
    // net_new shape: [1, num_edges, DIM] - updated net state
    // delta shape: [1, num_edges, 2] - flow correction
    // weight shape: [1, num_edges, 2] - confidence weights
    
    // Step 3: BA function (CPU-based, Eigen solver)
    // 3.1 Compute targets (reprojected coords + delta correction)
    auto target = compute_targets(coords, delta);
    // target shape: [1, num_edges, 2] - target coordinates
    pg_->target = target;
    pg_->weight = weight;
    
    // 3.2 Run bundle adjustment
    if (should_run_global_ba()) {
        run_global_ba();  // Global BA with all edges
    } else {
        run_bundle_adjustment(
            pg_->target,
            pg_->weight,
            pg_->ii,
            pg_->jj,
            pg_->kk);  // Local BA with active edges
    }
    
    // 3.3 Update point cloud from optimized poses and patches
    update_point_cloud();
}

// Step 1: Correlation implementation
Eigen::Tensor<float, 3> UpdateThread::compute_correlation(
    const Eigen::Tensor<float, 5>& coords) {
    
    // Get active edges
    int num_active = pg_->num_edges;
    std::vector<int> ii = pg_->ii;  // Source frames
    std::vector<int> jj = pg_->jj;  // Target frames
    std::vector<int> kk = pg_->kk;  // Patch indices
    
    // Get feature maps from state (circular buffers)
    auto gmap = state_->gmap_;  // Global patch features
    auto fmap1 = state_->fmap1_; // Full resolution feature map
    auto fmap2 = state_->fmap2_; // Downsampled feature map
    
    // Compute correlation at two scales (CPU-based, SIMD optimized)
    auto corr1 = correlation_cpu(gmap, fmap1, coords / 1, ii, jj, kk, 3);
    auto corr2 = correlation_cpu(gmap, fmap2, coords / 4, ii, jj, kk, 3);
    
    // Concatenate correlations
    return concatenate_correlations(corr1, corr2);
}

// Step 2: Update model inference
std::tuple<Eigen::Tensor<float, 3>, Eigen::MatrixXf, Eigen::MatrixXf>
UpdateThread::run_update_inference(
    const Eigen::Tensor<float, 3>& net,
    const Eigen::Tensor<float, 3>& ctx,
    const Eigen::Tensor<float, 3>& corr,
    const std::vector<int>& ii,
    const std::vector<int>& jj,
    const std::vector<int>& kk) {
    
    // Prepare inputs for update network
    // Convert Eigen tensors to eazyai input format
    std::vector<float> net_input = tensor_to_vector(net);
    std::vector<float> ctx_input = tensor_to_vector(ctx);
    std::vector<float> corr_input = tensor_to_vector(corr);
    
    // Run update network inference (EazyAI on CVflow)
    auto outputs = update_net_->run_update(net_input, ctx_input, corr_input,
                                           ii, jj, kk);
    
    // Convert outputs back to Eigen
    Eigen::Tensor<float, 3> net_new = vector_to_tensor(outputs.net, net.dimensions());
    Eigen::MatrixXf delta = vector_to_matrix(outputs.delta, {1, outputs.delta.size()/2, 2});
    Eigen::MatrixXf weight = vector_to_matrix(outputs.weight, {1, outputs.weight.size()/2, 2});
    
    return {net_new, delta, weight};
}

// Step 3: BA function
void UpdateThread::run_bundle_adjustment(
    const Eigen::MatrixXf& target,
    const Eigen::MatrixXf& weight,
    const std::vector<int>& ii,
    const std::vector<int>& jj,
    const std::vector<int>& kk) {
    
    // CPU-based bundle adjustment using Eigen
    int t0 = pg_->n - cfg_.OPTIMIZATION_WINDOW;
    t0 = std::max(t0, 1);
    int t1 = pg_->n;
    
    // Compute residuals and Hessian (CPU)
    Eigen::SparseMatrix<float> H;
    Eigen::VectorXf b;
    compute_ba_system(pg_->poses_, pg_->patches_, pg_->intrinsics_,
                     target, weight, ii, jj, kk, t0, t1, H, b);
    
    // Solve linear system (Eigen sparse solver)
    Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;
    solver.compute(H);
    Eigen::VectorXf delta_x = solver.solve(b);
    
    // Update poses and patches
    update_poses_and_patches(delta_x, t0, t1);
}
```

---

## 9. Memory Management

### 9.1 Pre-allocation Strategy

All buffers pre-allocated at initialization:
- PatchGraph: All tensors allocated with max size
- Feature maps: Circular buffers with fixed size
- Queues: Limited size to prevent memory growth

### 9.2 Memory Layout

```cpp
// Contiguous memory for cache efficiency
class AlignedBuffer {
    alignas(32) float* data;
    size_t size;
    
public:
    AlignedBuffer(size_t size) : size(size) {
        data = (float*)aligned_alloc(32, size * sizeof(float));
    }
    ~AlignedBuffer() { free(data); }
};
```

---

## 10. Performance Considerations

### 10.1 Bottleneck Analysis

**Typical Bottlenecks:**
1. **Thread 2 (Patchify)**: Network inference is usually slowest
2. **Thread 3 (Update)**: BA can be slow with many edges

**Optimization Strategies:**
- Use CV28's CVflow processor for neural network inference (via eazyai)
- Optimize BA with efficient CPU-based solvers (Eigen)
- Use SIMD instructions for correlation and projective operations
- Pre-allocate all buffers to avoid dynamic allocation
- Use INT8 quantization for models if supported by eazyai

### 10.2 Frame Rate Targets

- **Thread 1**: Should capture at sensor rate (e.g., 30 FPS)
- **Thread 2**: May be slower (network inference on CVflow), but pipeline helps
- **Thread 3**: Should process faster than Thread 2 to avoid queue buildup

**Ideal Scenario:**
- Thread 1: 30 FPS
- Thread 2: 20-25 FPS (network inference on CV28 CVflow processor)
- Thread 3: 25-30 FPS (optimization on CPU)

**CV28 Advantages:**
- CVflow processor handles neural network inference (hardware acceleration)
- CPU threads handle correlation and BA
- No CUDA needed - all operations are CPU-based or CVflow-accelerated

If Thread 3 is slower, queue will build up, but system still works.

---

## 11. Error Handling

### 11.1 Queue Timeouts

```cpp
// In Thread 2 and 3, use timeouts to detect stalls
FeatureFrame frame;
if (!feature_queue_->try_pop(frame, std::chrono::milliseconds(100))) {
    // Queue empty, but continue running
    continue;
}
```

### 11.2 Frame Dropping

If queue is full, drop oldest frames:
```cpp
if (queue_->size() >= max_size_) {
    RawFrame dropped;
    queue_->try_pop(dropped);  // Remove oldest
}
queue_->push(new_frame);
```

---

## 12. Build System for CV28

### 12.1 CMakeLists.txt (CV28 with EazyAI)

```cmake
cmake_minimum_required(VERSION 3.15)
project(DPVO_CV28)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CV28 cross-compilation toolchain
# Set CMAKE_TOOLCHAIN_FILE if cross-compiling
# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchain-cv28.cmake)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# EazyAI library (Ambarella CV28)
# Adjust paths based on your CV28 SDK installation
set(EAZYAI_ROOT "/path/to/cv28/sdk/eazyai")
set(EAZYAI_INCLUDE_DIR "${EAZYAI_ROOT}/include")
set(EAZYAI_LIB_DIR "${EAZYAI_ROOT}/lib")

include_directories(${EAZYAI_INCLUDE_DIR})
link_directories(${EAZYAI_LIB_DIR})

# Source files
add_library(dpvo_core STATIC
    src/dpvo/dpvo.cpp
    src/dpvo/patchgraph.cpp
    src/dpvo/thread_safe_queue.cpp
    src/dpvo/eazyai_inference.cpp  # EazyAI wrapper
    src/dpvo/correlation.cpp
    src/dpvo/bundle_adjustment.cpp
    src/dpvo/projective_ops.cpp
)

# Executable
add_executable(dpvo_cv28 src/main.cpp)
target_link_libraries(dpvo_cv28 
    dpvo_core
    Eigen3::Eigen
    opencv_core
    opencv_imgproc
    eazyai  # Ambarella eazyai library
    pthread
)

# CV28-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_compile_options(dpvo_cv28 PRIVATE 
        -march=armv8-a+simd
        -O3
        -ffast-math
    )
endif()

# Install models to CV28
install(FILES 
    models/fnet.bin
    models/inet.bin
    models/update.bin
    DESTINATION /opt/dpvo/models
)
```

### 12.2 CV28 Toolchain File (if cross-compiling)

```cmake
# toolchain-cv28.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# CV28 toolchain path
set(TOOLCHAIN_PREFIX "/path/to/cv28/toolchain")
set(CMAKE_C_COMPILER "${TOOLCHAIN_PREFIX}/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}/bin/aarch64-linux-gnu-g++")

set(CMAKE_FIND_ROOT_PATH 
    "${TOOLCHAIN_PREFIX}/aarch64-linux-gnu"
    "/path/to/cv28/sysroot"
)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

---

## 13. Testing Strategy

### 13.1 Unit Tests
- Thread-safe queue
- ONNX inference wrapper
- Individual components

### 13.2 Integration Tests
- 3-thread pipeline
- Frame ordering
- Queue overflow handling

### 13.3 Performance Tests
- Measure latency per thread
- Queue sizes
- Memory usage

---

## 14. Advantages of 3-Thread Design

✅ **Simplicity**: Easy to understand and debug
✅ **Pipeline Parallelism**: Natural overlap of computation
✅ **Low Overhead**: Minimal synchronization (just queues)
✅ **Predictable**: Clear data flow
✅ **Suitable for Embedded**: Low memory footprint, deterministic

---

## 15. CV28-Specific Considerations

### 15.1 Hardware Acceleration
- **CVflow Processor**: Neural network inference runs on dedicated CVflow hardware (via eazyai)
- **CPU**: Handles correlation, BA, and other operations
- **Memory**: CV28 has limited memory - pre-allocate buffers, limit queue sizes

### 15.2 Model Optimization
- Use Ambarella's model optimization tools
- Consider INT8 quantization for faster inference
- Optimize model input/output shapes for CV28

### 15.3 Performance Tuning
- Profile eazyai inference times
- Optimize CPU operations with SIMD
- Balance queue sizes to prevent memory issues
- Monitor CVflow processor utilization

### 15.4 Potential Improvements (Future)

1. **Dynamic Thread Pool**: Add more threads for BA if CPU cores available
2. **Priority Queue**: Process important frames first
3. **Adaptive Frame Dropping**: Drop frames based on motion
4. **Model Quantization**: Use INT8 models for faster inference
5. **Memory Pool**: Reuse buffers to reduce allocations

---

## Conclusion

**Yes, the 3-thread pipeline will work!** It's a clean, simple design that:
- Separates concerns clearly
- Provides natural parallelism
- Is suitable for embedded systems like CV28
- Avoids complex synchronization

The key is proper queue sizing and handling backpressure when Thread 3 is slower than Thread 2.

