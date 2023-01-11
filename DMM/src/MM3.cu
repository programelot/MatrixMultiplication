#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

__global__ void MM_Kernel(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY, size_t resUnit, size_t matUnit, size_t inputUnit){
    size_t global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = global/sizeY;
    size_t j = global%sizeY;
    global = i * resUnit + j;
    if(global > sizeX * sizeY) return;
    dataType acc = 0;
    for(int k = 0; k < sizeRange; ++k){
        acc +=  matrix[i * matUnit + k] * input[k * inputUnit + j];
    }
    res[global] = acc;
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
    HANDLE_ERROR(cudaMalloc(&input_GPU,  sizeof(int)*rowNumInp*colNumInp));
    HANDLE_ERROR(cudaMalloc(&res_GPU,    sizeof(int)*rowNumRes*colNumRes));
    HANDLE_ERROR(cudaMemcpy(matrix_GPU, matrix, sizeof(dataType) * colNumMat * rowNumMat, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(input_GPU, input, sizeof(dataType) * colNumInp * rowNumInp, cudaMemcpyHostToDevice));

    int work = rowNumRes * colNumRes;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel<<<(work + MAXTHREADSIZE - 1)/MAXTHREADSIZE, work > MAXTHREADSIZE? MAXTHREADSIZE : work >>>(matrix_GPU, input_GPU, res_GPU, rowSize, rangeSize, colSize, colSize, rangeSize, colSize);
    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * colNumRes * rowNumRes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}
