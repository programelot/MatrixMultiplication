#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

__global__ void MM_Kernel(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY, size_t resUnit, size_t matUnit, size_t inputUnit){
    size_t li = threadIdx.x; size_t bi = blockIdx.x; size_t bis = blockDim.x;
    size_t lj = threadIdx.y; size_t bj = blockIdx.y; size_t bjs = blockDim.y;
    size_t globalRowIdx = li + bi * bis;
    size_t globalColIdx = lj + bj * bjs;
    dataType acc = 0;
    __shared__ dataType subMatrix[subBlockSize][subBlockSize];
    __shared__ dataType subInput[subBlockSize][subBlockSize];
    for(int k = 0; k < (sizeRange + subBlockSize - 1)/subBlockSize; ++k){
        size_t localMatRowIdx = globalRowIdx;
        size_t localMatColIdx = lj + subBlockSize * k;
        size_t localInputRowIdx = li + subBlockSize * k;
        size_t localInputColIdx = globalColIdx;
        if(localMatRowIdx < sizeX && localMatColIdx < sizeRange) {
            subMatrix[li][lj] = matrix[localMatRowIdx * matUnit + localMatColIdx];
        }
        if(localInputRowIdx < sizeRange && localInputColIdx < sizeY){
            subInput[lj][li]  = input [localInputRowIdx * inputUnit + localInputColIdx];
        }
        __syncthreads();
        for(int lk = 0; lk < subBlockSize; ++lk){
            if(subBlockSize * k + lk < sizeRange){
                acc += subMatrix[li][lk] * subInput[lj][lk];
            }
        }
        __syncthreads();
    }
    if(globalRowIdx < sizeX && globalColIdx < sizeY)
        res[globalRowIdx * resUnit + globalColIdx] = acc;
}

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
    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType)*rowNumMat*colNumMat));
    HANDLE_ERROR(cudaMalloc(&input_GPU, sizeof(dataType)*rowNumInp*colNumInp));
    HANDLE_ERROR(cudaMalloc(&res_GPU, sizeof(dataType)*rowNumRes*colNumRes));
    HANDLE_ERROR(cudaMemcpy(matrix_GPU, matrix, sizeof(dataType) * colNumMat * rowNumMat, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(input_GPU, input, sizeof(dataType) * colNumInp * rowNumInp, cudaMemcpyHostToDevice));

    dim3 threadDim(rowNumRes > subBlockSize ? subBlockSize : rowNumRes , colNumRes > subBlockSize ? subBlockSize : colNumRes);
    dim3 blockDim((rowNumRes + subBlockSize - 1)/subBlockSize , (colNumRes + subBlockSize - 1)/subBlockSize);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel<<<blockDim, threadDim>>>(matrix_GPU, input_GPU, res_GPU, rowSize, rangeSize, colSize, colSize, rangeSize, colSize);
    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * colNumRes * rowNumRes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}
