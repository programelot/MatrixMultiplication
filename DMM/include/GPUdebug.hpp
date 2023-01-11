#include <iostream>
#include <cublas.h>

#ifndef GPUDEBUG
#define GPUDEBUG

static const char* cublasGetErrorString(cublasStatus_t status){
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cout << "Cuda Error : " << cudaGetErrorString( err ) << " in " << file << " at line " << line << std::endl;
        exit(0);
    }
}

static void HandleCublasError(cublasStatus_t err, const char *file, int line){
    if (err != CUBLAS_STATUS_SUCCESS){
        std::cout << "CUBLAS Error : " << cublasGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(0);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define HANDLE_CUBLAS_ERROR( err ) (HandleCublasError( err, __FILE__, __LINE__ ))

#endif