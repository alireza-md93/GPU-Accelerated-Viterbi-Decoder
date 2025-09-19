#ifndef __GPU_ERRORS_H__
#define __GPU_ERRORS_H__

#include <iostream>
#include <cuda_runtime.h>

#if __cplusplus >= 201703L
inline void HandleError(cudaError_t err, const char *file, int line) {
#else
static void HandleError(cudaError_t err, const char *file, int line) {
#endif
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
// #define HANDLE_ERROR( err ) 1


#define HANDLE_NULL( a ) {if ((a) == nullptr) { \
                            std::cerr << "Host memory failed in " << __FILE__ << " at line " << __LINE__ << std::endl; \
                            std::exit(EXIT_FAILURE);}}
							
#endif  /* __GPU_ERRORS_H__ */