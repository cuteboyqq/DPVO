# C++ DPVO for Ambarella CV28 - File Tree

```
cpp_dpvo_cv28/
│
├── CMakeLists.txt                          # Main CMake build file
├── README.md                               # Project documentation
├── toolchain-cv28.cmake                    # CV28 cross-compilation toolchain
│
├── include/                                 # Header files
│   └── dpvo/
│       ├── dpvo.hpp                        # Main DPVO class
│       ├── config.hpp                      # Configuration management
│       ├── types.hpp                       # Common types and enums
│       ├── thread_safe_queue.hpp           # Thread-safe queue implementation
│       │
│       ├── threads/                        # Thread classes
│       │   ├── image_acquisition_thread.hpp
│       │   ├── patchify_thread.hpp
│       │   └── update_thread.hpp
│       │
│       ├── network/                        # Neural network inference
│       │   ├── eazyai_inference.hpp        # EazyAI wrapper
│       │   └── model_loader.hpp            # Model loading utilities
│       │
│       ├── patchgraph/                     # Data structures
│       │   ├── patchgraph.hpp              # PatchGraph class
│       │   └── dpvo_state.hpp               # DPVO state management
│       │
│       ├── correlation/                    # Correlation operations
│       │   ├── correlation.hpp             # Correlation engine
│       │   └── correlation_cpu.hpp         # CPU-based correlation kernels
│       │
│       ├── ba/                             # Bundle adjustment
│       │   ├── bundle_adjustment.hpp        # BA solver
│       │   ├── ba_solver.hpp               # Linear system solver
│       │   └── ba_residuals.hpp            # Residual computation
│       │
│       ├── projective/                     # Projective operations
│       │   ├── projective_ops.hpp          # Transform, reproject, etc.
│       │   └── se3.hpp                     # SE3 Lie group operations
│       │
│       └── utils/                          # Utility functions
│           ├── patchify_ops.hpp            # Patch extraction utilities
│           ├── image_utils.hpp             # Image preprocessing
│           └── tensor_utils.hpp          # Eigen tensor utilities
│
├── src/                                     # Source files
│   ├── main.cpp                            # Main entry point
│   │
│   └── dpvo/
│       ├── dpvo.cpp                        # Main DPVO implementation
│       ├── config.cpp                      # Configuration loading
│       ├── thread_safe_queue.cpp           # Queue implementation
│       │
│       ├── threads/
│       │   ├── image_acquisition_thread.cpp
│       │   ├── patchify_thread.cpp
│       │   └── update_thread.cpp
│       │
│       ├── network/
│       │   ├── eazyai_inference.cpp        # EazyAI wrapper implementation
│       │   └── model_loader.cpp
│       │
│       ├── patchgraph/
│       │   ├── patchgraph.cpp
│       │   └── dpvo_state.cpp
│       │
│       ├── correlation/
│       │   ├── correlation.cpp
│       │   └── correlation_cpu.cpp        # CPU correlation kernels
│       │
│       ├── ba/
│       │   ├── bundle_adjustment.cpp
│       │   ├── ba_solver.cpp
│       │   └── ba_residuals.cpp
│       │
│       ├── projective/
│       │   ├── projective_ops.cpp
│       │   └── se3.cpp
│       │
│       └── utils/
│           ├── patchify_ops.cpp
│           ├── image_utils.cpp
│           └── tensor_utils.cpp
│
├── models/                                 # CV28 model files
│   ├── fnet.bin                            # Feature network (CV28 format)
│   ├── inet.bin                            # Instance network (CV28 format)
│   ├── update.bin                          # Update network (CV28 format)
│   │
│   └── README.md                           # Model conversion instructions
│
├── config/                                 # Configuration files
│   ├── default.yaml                        # Default configuration
│   ├── fast.yaml                          # Fast mode configuration
│   └── cv28.yaml                          # CV28-specific configuration
│
├── tests/                                  # Unit tests
│   ├── CMakeLists.txt
│   ├── test_thread_safe_queue.cpp
│   ├── test_correlation.cpp
│   ├── test_bundle_adjustment.cpp
│   ├── test_patchify.cpp
│   └── test_eazyai_inference.cpp
│
├── examples/                               # Example programs
│   ├── simple_demo.cpp                    # Simple usage example
│   ├── video_demo.cpp                     # Video processing example
│   └── sensor_demo.cpp                    # Sensor input example
│
├── scripts/                               # Build and utility scripts
│   ├── build_cv28.sh                      # Build script for CV28
│   ├── convert_models.sh                 # Model conversion script
│   └── deploy_to_cv28.sh                 # Deployment script
│
├── thirdparty/                            # Third-party dependencies
│   ├── eigen/                             # Eigen library (if not system-installed)
│   └── README.md                          # Third-party setup instructions
│
└── docs/                                  # Documentation
    ├── architecture.md                    # Architecture overview
    ├── api_reference.md                   # API documentation
    ├── build_instructions.md             # Build instructions
    ├── model_conversion.md                # Model conversion guide
    └── performance_tuning.md              # Performance optimization guide
```

## Key Files Description

### Core Files
- **`include/dpvo/dpvo.hpp`**: Main DPVO orchestrator class
- **`include/dpvo/config.hpp`**: Configuration management
- **`include/dpvo/thread_safe_queue.hpp`**: Thread-safe queue for inter-thread communication

### Thread Files
- **`include/dpvo/threads/image_acquisition_thread.hpp`**: Thread 1 - Image capture
- **`include/dpvo/threads/patchify_thread.hpp`**: Thread 2 - Feature extraction
- **`include/dpvo/threads/update_thread.hpp`**: Thread 3 - Optimization

### Network Files
- **`include/dpvo/network/eazyai_inference.hpp`**: EazyAI wrapper for CV28 models

### Core Algorithms
- **`include/dpvo/correlation/correlation.hpp`**: Correlation computation
- **`include/dpvo/ba/bundle_adjustment.hpp`**: Bundle adjustment solver
- **`include/dpvo/projective/projective_ops.hpp`**: Projective transformations

### Data Structures
- **`include/dpvo/patchgraph/patchgraph.hpp`**: PatchGraph data structure
- **`include/dpvo/patchgraph/dpvo_state.hpp`**: DPVO state management

## Build Structure

```
build/                                      # Build directory (created by CMake)
├── CMakeCache.txt
├── CMakeFiles/
├── cmake_install.cmake
├── Makefile
├── libdpvo_core.a                         # Static library
└── dpvo_cv28                              # Executable
```

## Installation Structure (on CV28)

```
/opt/dpvo/                                 # Installation directory
├── bin/
│   └── dpvo_cv28                          # Executable
├── lib/
│   └── libdpvo_core.a                     # Library (if needed)
├── models/
│   ├── fnet.bin
│   ├── inet.bin
│   └── update.bin
└── config/
    ├── default.yaml
    └── cv28.yaml
```

## File Count Summary

- **Headers**: ~20 files
- **Sources**: ~20 files
- **Models**: 3 files (.bin)
- **Config**: 3 files (.yaml)
- **Tests**: ~5 files
- **Examples**: ~3 files
- **Scripts**: ~3 files
- **Docs**: ~5 files

**Total**: ~60+ files

## Minimal Structure (MVP)

For a minimal viable implementation, you can start with:

```
cpp_dpvo_cv28/
├── CMakeLists.txt
├── include/
│   └── dpvo/
│       ├── dpvo.hpp
│       ├── config.hpp
│       ├── thread_safe_queue.hpp
│       ├── threads/
│       │   ├── image_acquisition_thread.hpp
│       │   ├── patchify_thread.hpp
│       │   └── update_thread.hpp
│       ├── network/
│       │   └── eazyai_inference.hpp
│       ├── patchgraph/
│       │   └── patchgraph.hpp
│       ├── correlation/
│       │   └── correlation.hpp
│       ├── ba/
│       │   └── bundle_adjustment.hpp
│       └── projective/
│           └── projective_ops.hpp
├── src/
│   ├── main.cpp
│   └── dpvo/
│       ├── dpvo.cpp
│       ├── config.cpp
│       ├── thread_safe_queue.cpp
│       ├── threads/
│       │   ├── image_acquisition_thread.cpp
│       │   ├── patchify_thread.cpp
│       │   └── update_thread.cpp
│       ├── network/
│       │   └── eazyai_inference.cpp
│       ├── patchgraph/
│       │   └── patchgraph.cpp
│       ├── correlation/
│       │   └── correlation.cpp
│       ├── ba/
│       │   └── bundle_adjustment.cpp
│       └── projective/
│           └── projective_ops.cpp
└── models/
    ├── fnet.bin
    ├── inet.bin
    └── update.bin
```

This minimal structure has ~25 files and covers all essential functionality.

