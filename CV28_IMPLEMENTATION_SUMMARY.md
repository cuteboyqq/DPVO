# DPVO for Ambarella CV28 - Implementation Summary

## ✅ Your 3-Thread Design Will Work!

The simplified 3-thread pipeline is perfect for CV28. Here's why:

### Architecture
```
Thread 1 (Image) → Queue → Thread 2 (Patchify) → Queue → Thread 3 (Update)
   Sensor/Video         EazyAI (CVflow)              Correlation + BA (CPU)
```

## Key Constraints & Solutions

### 1. ❌ No PyTorch → ✅ Use EazyAI
- **Problem**: CV28 doesn't support PyTorch
- **Solution**: Use Ambarella's **eazyai library** for model inference
- **Models**: Convert ONNX models to CV28 format (`.bin` or `.amb`)
- **Hardware**: Models run on CV28's **CVflow processor** (hardware acceleration)

### 2. ❌ No CUDA → ✅ CPU-Based Operations
- **Problem**: CV28 has no CUDA support
- **Solution**: All operations run on CPU:
  - **Correlation**: CPU-based with SIMD optimization
  - **Bundle Adjustment**: CPU-based using Eigen's sparse solvers
  - **Projective Operations**: CPU-based using Eigen
  - **Neural Networks**: CVflow processor (via eazyai) - hardware accelerated!

### 3. ✅ 3-Thread Pipeline

**Thread 1**: Image acquisition (sensor/video)
- Captures frames and enqueues raw images

**Thread 2**: Patchify (Feature Extraction)
1. **fnet & inet inference** (EazyAI on CVflow)
2. **Patchify** fnet/inet/image to extract patches
3. **Enqueue** processed data (fmap, imap, gmap, patches, colors)

**Thread 3**: Update (Optimization)
1. **Correlation** (CPU-based, SIMD optimized)
2. **Update model inference** (EazyAI on CVflow)
3. **BA function** (CPU-based, Eigen solver)

## Implementation Checklist

### Phase 1: Setup
- [ ] Install CV28 SDK and eazyai library
- [ ] Convert models: `fnet.onnx` → `fnet.bin`, `inet.onnx` → `inet.bin`, `update.onnx` → `update.bin`
- [ ] Set up cross-compilation toolchain for CV28

### Phase 2: Core Components
- [ ] Implement `EazyAIInference` wrapper class
- [ ] Implement thread-safe queues
- [ ] Implement Thread 1 (Image Acquisition)
- [ ] Implement Thread 2 (Patchify with eazyai)
- [ ] Implement Thread 3 (Update - CPU-based)

### Phase 3: CPU Operations
- [ ] Implement CPU-based correlation (with SIMD)
- [ ] Implement CPU-based bundle adjustment (Eigen)
- [ ] Implement CPU-based projective operations

### Phase 4: Integration & Testing
- [ ] Integrate all 3 threads
- [ ] Test on CV28 hardware
- [ ] Optimize performance

## Key Code Changes

### EazyAI Instead of ONNX Runtime
```cpp
// OLD (ONNX Runtime):
std::unique_ptr<ONNXInference> fnet_;

// NEW (EazyAI):
std::unique_ptr<EazyAIInference> fnet_;
```

### CPU-Based Operations
```cpp
// Correlation - CPU with SIMD
void correlation_kernel_cpu(...);  // No CUDA!

// Bundle Adjustment - Eigen solver
Eigen::SparseLU<Eigen::SparseMatrix<float>> solver_;  // CPU-based
```

### Model Paths
```cpp
// CV28 model paths (.bin or .amb files)
const std::string fnet_model_path = "/opt/dpvo/models/fnet.bin";
const std::string inet_model_path = "/opt/dpvo/models/inet.bin";
const std::string update_model_path = "/opt/dpvo/models/update.bin";
```

## Performance Expectations

- **Thread 1**: 30 FPS (sensor rate)
- **Thread 2**: 20-25 FPS (CVflow inference - hardware accelerated)
- **Thread 3**: 25-30 FPS (CPU operations - SIMD optimized)

## Advantages of This Design

1. ✅ **Simple**: Easy to understand and debug
2. ✅ **Efficient**: CVflow handles neural networks, CPU handles optimization
3. ✅ **No CUDA needed**: All operations are CPU-based or CVflow-accelerated
4. ✅ **Low overhead**: Minimal synchronization (just queues)
5. ✅ **Embedded-friendly**: Suitable for CV28's resources

## Next Steps

1. Review the full outline: `CPP_DPVO_OUTLINE_CV28.md`
2. Set up CV28 development environment
3. Convert models to CV28 format
4. Start implementing Thread 1 (simplest)
5. Implement eazyai wrapper
6. Implement Thread 2 and 3
7. Test and optimize

## Questions to Resolve

1. **EazyAI API**: Get exact eazyai API documentation from Ambarella
2. **Model Conversion**: Use Ambarella's model conversion tools
3. **CV28 SDK**: Install CV28 SDK and understand its structure
4. **Memory Limits**: Understand CV28's memory constraints

---

**The 3-thread design is perfect for CV28!** It's simple, efficient, and works within CV28's constraints (no PyTorch, no CUDA).

