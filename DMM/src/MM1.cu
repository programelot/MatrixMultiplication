#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

float gemm(dataType* matrix, dataType* input, dataType* res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize){
    const size_t rowNumMat = rowSize;
    const size_t rowNumInp = rangeSize;
    const size_t rowNumRes = rowSize;
    const size_t colNumMat = rangeSize;
    const size_t colNumInp = colSize;
    const size_t colNumRes = colSize;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    for(int i = 0; i < rowNumRes; ++i){
        for(int j = 0; j < colNumRes; ++j){
            res[i * rowSize + j] = 0;
            for(int k = 0; k < colNumMat; ++k){
                res[i * rowSize + j] += matrix[i * rowSize + k] * input[k * rangeSize + j];
            }
        }
    }
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}
