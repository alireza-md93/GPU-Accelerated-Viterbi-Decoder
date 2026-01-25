# High-Performance CUDA Viterbi Decoder

This project provides a high-performance Viterbi decoder accelerated in CUDA, suitable for high-throughput GPU-based Software Defined Radio (SDR) systems. By leveraging modern CUDA features and novel algorithmic approaches, it can achieve throughputs of **over 100 Gb/s** on consumer-grade GPUs.

In previous works on GPU-based Viterbi decoding, implementers have often used block-wise approaches where the input batch is divided into small, overlapping chunks. Furthermore, traceback operations typically required a separate kernel launch and allocated a very large amount of global memory to store traceback pointers. This implementation overcomes those limitations.

## Key Innovations

### 1. Optimized Add-Compare-Select (ACS) Pass
- **Warp-Level Parallelism**: The core ACS butterfly operation is implemented using warp-level shuffling intrinsics (`__shfl_xor_sync`). This eliminates synchronization overhead and avoids slow shared memory access for path metric exchange between threads.
- **SIMD Intrinsics**: Computational throughput is boosted by leveraging low-level DPX SIMD intrinsics. The implementation uses `int16x2` operations for GPUs with Compute Capability 9.0+ and `int32` for broad compatibility with modern NVIDIA GPUs.
- **Balanced Compute Pipeline**: Leverages `__half2` metrics to offload path metric calculations to FP16/Tensor Cores, while path selection logic remains on the ALUs. This balances the workload between different computation pipelines, preventing ALU overloading and maximizing throughput.

### 2. Ultrafast & Memory-Efficient Traceback
- **Register Exchange Algorithm**: This implementation revives the often-neglected register exchange algorithm. By storing path histories directly in the fast on-chip registers of each thread, it enables an ultrafast traceback that avoids costly memory operations.
- **Kernel Fusion**: A single, unified kernel performs both the forward ACS pass and the backward traceback pass. While the traceback is performed by a single thread (creating divergence), the overhead is confidently considered negligible because of the ultrafast traceback. It also reduces kernel launch latency and, crucially, enables the one-pointer algorithm.
- **Persistent Kernel Design**: The kernel is designed to be persistent, which can be viewed as a block-wise decoder with an extremely large block length. This strategy maximizes data locality and improves L2 cache utilization, keeping the GPU constantly fed with work.
- **One-Pointer Traceback**: The unified kernel design allows traceback data in global memory to be overwritten as soon as it's consumed. This enables a "one-pointer" circular buffer strategy, which combined with the persistent kernel design, drastically reduces the memory footprint for storing traceback paths by over **1000x** compared to conventional methods.

## Core Features

- **Optimized for Multiple GPU Architectures**:
    - Kernels based on **`int32` DPX intrinsics**, providing high performance on all modern NVIDIA GPUs.
    - Highly-optimized kernels using **`int16` SIMD intrinsics**, which can improve performance on GPUs with Compute Capability 9.0 or higher (e.g., Ada Lovelace, Hopper).
    - Kernels leveraging the `__half2` data type for metrics and `uint16_t` for decisions to balance the computational load across multiple pipelines.

- **Flexible Input Types**: The decoder supports both hard and soft-decision inputs with varying precision levels. For all soft-decision types, a negative value is interpreted as a '0' bit and a positive value as a '1' bit.
  - **Hard Decision**: Input type is `int32`, where each element contains 32 packed encoded bits from the channel.
  - **4-bit Soft-Decision**: Input type is `int32`, with each element containing eight packed 4-bit soft-decision values.
  - **8-bit Soft-Decision**: Input type is `int32`, with each element containing four packed 8-bit soft-decision values.
  - **16-bit Soft-Decision**: Input type is `int32`, with each element containing two packed 16-bit soft-decision values.
  - **Floating-Point Soft-Decision**: Input type is `float`, where each element represents a single soft-decision value.

For all packed formats, the Most Significant Bit (MSB) corresponds to the input received earliest in time.

## How to Run a Test

### Prerequisites

*   An NVIDIA GPU with CUDA support.
*   CUDA Toolkit (11.0 or newer recommended).
*   A C++17 compliant compiler (e.g., GCC, Clang, MSVC).

### Compilation

The project can be compiled using `nvcc`, the NVIDIA CUDA Compiler. A Makefile is available to facilitate the compilation. Open a terminal and run `make` from the root directory of the project. Modify compiler flags as needed.

### Execution

The compiled executable runs a simulation pipeline that generates random bits, encodes them, adds noise, and finally decodes them to measure the Bit Error Rate (BER).

```bash
# Run the simulation with default parameters
./main

# Run with a custom parameters
./main -n 1000000 -s 5.5 -m b32 -i s4
```

**Command-Line Options:**

- **`-n, --num <integer>`**  
  Number of bits in the original message.  
  *Default:* `32,000,000`

- **`-s, --snr <float>`**  
  Signal-to-Noise Ratio (SNR) in dB.  
  *Default:* `15.0`

  - **`-i, --input <type>`**  
  Channel input type. Accepted values:  
  - `HARD` or `h` (Default)
  - `SOFT4` or `s4`  
  - `SOFT8` or `s8`  
  - `SOFT16` or `s16`  
  - `FP32` or `f`

- **`-m, --metric <type>`**  
  Metric type used in the decoder core. Accepted values:  
  - `b16` → 16-bit core  
  - `b32` → 32-bit core (Default)
  - `f16` → FP16 core 

- **`-o, --output <type>`**  
  Integer type used to store packed output bits. Accepted values:  
  - `b16` → An array of 16-bit unsigned integers, each containing 16 output bits (latest bit in LSB). 
  - `b32` → An array of 32-bit unsigned integers, each containing 32 output bits (latest bit in LSB). (Default)

- **`-c, --compMode <type>`**  
  Computation mode for the decoder core. Accepted values:  
  - `dpx` → Enables DPX intrinsics (requires Compute Capability 9.0+). Only available with integer metrics (`b16`, `b32`).
  - `reg` → Regular computation core. (Default)


- **`-v, --verbose`** 
  Show more details.

- **`-h, --help`**  
  Display the help message.

  ## How to Use the API

  The core decoder is encapsulated within the `ViterbiCUDA` class located in the `viterbi` directory. To use the decoder, you must instantiate this class with specific template parameters that define the processing core and input data type.
  
  In order to avoid compiling all combinations of template parameters, modify the `OptionsValid` struct in `viterbi/viterbi.h`. This struct receives the same template parameter as `ViterbiCUDA`, and its boolean member `value` should be true for desired options. It is already set false for invalid cases.

  ### Template Parameters
  The `ViterbiCUDA` class is templated as `ViterbiCUDA<int options>`. A specific configuration can be passed as a bitwise OR of parameters defined as enumerations. The following are the parameters to be configured:
  
  - **`ChannelIn`**: Defines the input data type from the channel (`encPack_t`). For soft-decision types, negative values are treated as a '0' bit and positive values as a '1' bit.
    - **`ChannelIn::HARD`**: `int32_t` containing 32 packed 1-bit hard-decision values.
    - **`ChannelIn::SOFT4`**: `int32_t` containing 8 packed 4-bit soft-decision values.
    - **`ChannelIn::SOFT8`**: `int32_t` containing 4 packed 8-bit soft-decision values.
    - **`ChannelIn::SOFT16`**: `int32_t` containing 2 packed 16-bit soft-decision values.
    - **`ChannelIn::FP32`**: `float` representing a single soft-decision value.

  - **`Metric`**: Defines the data type for branch and path metrics (`metric_t`).
    - **`Metric::M_B16`**: Uses `int16_t` for `metric_t`. This enables `int16x2` SIMD-based DPX intrinsics for GPUs with Compute Capability 9.0+.
    - **`Metric::M_B32`**: Uses `int32_t` for `metric_t`. This uses standard 32-bit DPX intrinsics for broader GPU compatibility.
    - **`Metric::M_FP16`**: Uses `__half` for `metric_t` and `uint16_t` for `decPack_t`. This uses FP16/Tensor cores for metric calculations.

  - **`DecodeOut`**: Defines the data type for the packed decoded output (`decPack_t`).
    - **`DecodeOut::O_B16`**: Uses `uint16_t` for `decPack_t`.
    - **`DecodeOut::O_B32`**: Uses `uint32_t` for `decPack_t`.

  - **`CompMode`**: Defines the computation mode.
    - **`CompMode::DPX`**: This enables `int16x2` SIMD-based DPX intrinsics for 16-bit metrics and `int32` DPX intrinsics for 32-bit metrics on GPUs with Compute Capability 9.0+.
    - **`CompMode::REG`**: Uses regular computation mode.
  

  Invalid configurations:
  - `Metric::M_B16` `| ChannelIn::SOFT16` 
  - `Metric::M_FP16` `| ChannelIn::SOFT16`
  - `Metric::M_FP16` `| ChannelIn::SOFT8`
  - `Metric::M_FP16` `| CompMode::DPX`



  
  ### Class Interface
  
  #### Constructor
  ```cpp
  ViterbiCUDA();
  ViterbiCUDA(size_t inputNum);
  ```
  - The default constructor prepares the decoder but allocates/deallocates memory on each `run()` call.
  - The constructor with `inputNum` pre-allocates device memory for a fixed input size, which is more efficient for repeated calls with same-sized data. `inputNum` is the total number of encoded bits (e.g., for `SOFT8`, it is the length of the input array multiplied by 4).
  
  #### Main Execution
  ```cpp
  void run(encPack_t* input_h, decPack_t* output_h, size_t inputNum, float* kernelTime = nullptr);
  ```
  - Decodes the `input_h` data from the host and writes the result to the `output_h` host pointer.
  - A few bits at the start (`extraL`) and end (`extraR`) of the output stream are omitted due to the nature of the algorithm.
  - If a pointer is provided for `kernelTime`, it will be populated with the kernel execution time in milliseconds.
  
  #### Utility Functions
  ```cpp
  size_t getInputSize(size_t inputNum);
  ```
  - Calculates the required input buffer size in bytes for a given number of encoded bits (`inputNum`).
  
  ```cpp
  size_t getMessageLen(size_t inputNum);
  ```
  - Calculates the number of decoded output bits that will be produced. This accounts for the discarded bits from the beginning and end of the stream.
  
  ```cpp
  size_t getOutputSize(size_t inputNum);
  ```
  - Calculates the required output buffer size in bytes.

  ### Example Usage
  ```cpp
  #include "viterbi.h"
  
  // Instantiate a decoder for 8-bit soft-decision inputs and a 32-bit metric core
  using ViterbiType = ViterbiCUDA<ChannelIn::SOFT8 | Metric::M_B32 | DecodeOut::O_B32>;
  ViterbiType decoder;
  
  size_t num_encoded_bits = 1000000;
  size_t input_size_bytes = decoder.getInputSize(num_encoded_bits);
  size_t output_size_bytes = decoder.getOutputSize(num_encoded_bits);
  
  // Allocate host memory
  auto host_input = new ViterbiType::encPack_t[input_size_bytes / sizeof(ViterbiType::encPack_t)];
  auto host_output = new ViterbiType::decPack_t[output_size_bytes / sizeof(ViterbiType::decPack_t)];
  
  // ... fill host_input with data ...
  
  float elapsed_time;
  decoder.run(host_input, host_output, num_encoded_bits, &elapsed_time);
  ```
