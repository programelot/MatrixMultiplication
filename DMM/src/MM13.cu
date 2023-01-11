#include "GPUdebug.hpp"
#include "common.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

float gemm(dataType* matrix, dataType* input, dataType* res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize){
    const size_t rowNumMat = rowSize;
    const size_t rowNumInp = rangeSize;
    const size_t rowNumRes = rowSize;
    const size_t colNumMat = rangeSize;
    const size_t colNumInp = colSize;
    const size_t colNumRes = colSize;
    dataType *matrix_GPU;
    dataType *input_GPU;
    dataType *res_GPU;
    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType) * rowNumMat * colNumMat));
    HANDLE_ERROR(cudaMalloc(&input_GPU, sizeof(dataType)  * rowNumInp * colNumInp));
    HANDLE_ERROR(cudaMalloc(&res_GPU, sizeof(dataType)    * rowNumRes * colNumRes));
    HANDLE_ERROR(cudaMemcpy(matrix_GPU, matrix, sizeof(dataType) * rowNumMat * colNumMat, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(input_GPU,  input,  sizeof(dataType) * rowNumInp * colNumInp, cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    HANDLE_ERROR(cudaDeviceSynchronize());
    float alpha = 1, beta = 0;

    cublasHandle_t handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&handle));
    
    HANDLE_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    colSize, rowSize, rangeSize, &alpha, input_GPU, colNumInp, matrix_GPU, colNumMat, &beta, res_GPU, colNumRes)); 
    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * colNumRes * rowNumRes, cudaMemcpyDeviceToHost));
    HANDLE_CUBLAS_ERROR(cublasDestroy(handle));

    HANDLE_ERROR(cudaDeviceSynchronize());
    
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}
