# High-Performance CUDA Viterbi Decoder

This project provides a high-performance Viterbi decoder accelerated in CUDA, suitable for high-throughput GPU-based Software Defined Radio (SDR) systems. By leveraging modern CUDA features and novel algorithmic approaches, it can achieve throughputs of **over 100 Gb/s** on consumer-grade GPUs.

In previous works on GPU-based Viterbi decoding, implementers have often used block-wise approaches where the input batch is divided into small, overlapping chunks. Furthermore, traceback operations typically required a separate kernel launch and allocated a very large amount of global memory to store traceback pointers. This implementation overcomes those limitations.

## Key Innovations

### 1. Optimized Add-Compare-Select (ACS) Pass
- **Warp-Level Parallelism**: The core ACS butterfly operation is implemented using warp-level shuffling intrinsics (`__shfl_xor_sync`). This eliminates synchronization overhead and avoids slow shared memory access for path metric exchange between threads.
- **SIMD Intrinsics**: Computational throughput is boosted by leveraging low-level DPX SIMD intrinsics. The implementation uses `int16x2` operations for GPUs with Compute Capability 9.0+ and `int32` for broad compatibility with modern NVIDIA GPUs.

### 2. Ultrafast & Memory-Efficient Traceback
- **Register Exchange Algorithm**: This implementation revives the often-neglected register exchange algorithm. By storing path histories directly in the fast on-chip registers of each thread, it enables an ultrafast traceback that avoids costly memory operations.
- **Kernel Fusion**: A single, unified kernel performs both the forward ACS pass and the backward traceback pass. While the traceback is performed by a single thread (creating divergence), the overhead is confidently considered negligible because of the ultrafast traceback. It also reduces kernel launch latency and, crucially, enables the one-pointer algorithm.
- **Persistent Kernel Design**: The kernel is designed to be persistent, which can be viewed as a block-wise decoder with an extremely large block length. This strategy maximizes data locality and improves L2 cache utilization, keeping the GPU constantly fed with work.
- **One-Pointer Traceback**: The unified kernel design allows traceback data in global memory to be overwritten as soon as it's consumed. This enables a "one-pointer" circular buffer strategy, which combined with the persistent kernel design, drastically reduces the memory footprint for storing traceback paths by over **1000x** compared to conventional methods.

## Core Features

- **Optimized for Multiple GPU Architectures**:
    - A kernel based on **`int32` DPX intrinsics**, providing high performance on all modern NVIDIA GPUs.
    - A highly-optimized kernel using **`int16` SIMD intrinsics**, which can nearly double the performance on GPUs with Compute Capability 9.0 or higher (e.g., Ada Lovelace, Hopper).

- **Flexible Input Types**: The decoder supports both hard and soft-decision inputs with varying precision levels. For all soft-decision types, a negative value is interpreted as a '0' bit and a positive value as a '1' bit.
  - **Hard Decision**: Input type is `int32`, where each element contains 32 packed encoded bits from the channel.
  - **4-bit Soft-Decision**: Input type is `int32`, with each element containing eight packed 4-bit soft-decision values.
  - **8-bit Soft-Decision**: Input type is `int32`, with each element containing four packed 8-bit soft-decision values.
  - **16-bit Soft-Decision**: Input type is `int32`, with each element containing two packed 16-bit soft-decision values.
  - **Floating-Point Soft-Decision**: Input type is `float`, where each element represents a single soft-decision value.

For all packed formats, the Most Significant Bit (MSB) corresponds to the input received earliest in time.

## How to Use

### Prerequisites

*   An NVIDIA GPU with CUDA support.
*   CUDA Toolkit (11.0 or newer recommended).
*   A C++17 compliant compiler (e.g., GCC, Clang, MSVC).

### Compilation

The project can be compiled using `nvcc`, the NVIDIA CUDA Compiler. A Makefile is avialble to facilitate the compilation. Open a terminal and run `make` from the root directory of the project. Modify compiler flags as needed.

### Execution

The compiled executable runs a simulation pipeline that generates random bits, encodes them, adds noise, and finally decodes them to measure the Bit Error Rate (BER).

```bash
# Run the simulation with default parameters
./main

# Run with a custom parameters
./main -n 1000000 -s 5.5 -m 32 -i s4
```

**Command-Line Options:**

- **`-n, --num <integer>`**  
  Number of bits in the original message.  
  *Default:* `32,000,000`

- **`-s, --snr <float>`**  
  Signal-to-Noise Ratio (SNR) in dB.  
  *Default:* `15.0`

- **`-m, --metric <type>`**  
  Metric type used in the decoder core. Accepted values:  
  - `16` or `16bit` → 16-bit core  
  - `32` or `32bit` → 32-bit core (Default)

- **`-i, --input <type>`**  
  Channel input type. Accepted values:  
  - `HARD` or `h` (Default)
  - `SOFT4` or `s4`  
  - `SOFT8` or `s8`  
  - `SOFT16` or `s16`  
  - `FP32` or `f`

- **`-h, --help`**  
  Display the help message.