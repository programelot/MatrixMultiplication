#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <time.h>
#include "common.hpp"
#include "GPUdebug.hpp"
using namespace std;

__global__ void MM_Kernel__correct(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY){
    size_t global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = global/sizeY;
    size_t j = global%sizeY;
    global = i * sizeY + j;
    if(global > sizeX * sizeY) return;
    res[global] = 0;
    for(int k = 0; k < sizeRange; ++k){
        res[global] += matrix[i * sizeRange + k] * input[k * sizeY + j];
    }
}

float MM_Correct(dataType* matrix, dataType* input, dataType* res,
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

    int work = rowNumRes * colNumRes;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));
    MM_Kernel__correct<<<(work + MAXTHREADSIZE - 1)/MAXTHREADSIZE, work > MAXTHREADSIZE? MAXTHREADSIZE : work >>>(matrix_GPU, input_GPU, res_GPU, rowSize, rangeSize, colSize);
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * colNumRes * rowNumRes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}

extern float gemm(dataType* matrix, dataType* input, dataType* res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize);

dataType randomF(){
    return (((double)rand())/RAND_MAX) * 2 - 1;
}

int main(int argc, char **argv){
    srand(time(NULL));

    if(argc != 4)
        return 0;

    size_t sizeX, sizeRange, sizeY;
    sizeX = 0;
    sizeY = 0;
    sizeRange = 0;
    
    char* v = argv[1];
    while(*v != '\0'){
        sizeX *= 10;
        sizeX += *v - '0';
        ++v;
    }
    
    v = argv[2];
    while(*v != '\0'){
        sizeRange *= 10;
        sizeRange += *v - '0';
        ++v;
    }

    v = argv[3];
    while(*v != '\0'){
        sizeY *= 10;
        sizeY += *v - '0';
        ++v;
    }

    dataType *matrix = new dataType[sizeX * sizeRange];
    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeRange; ++j){
            matrix[i * sizeRange + j] = randomF();
        }
    }
    
    dataType *input = new dataType[sizeRange * sizeY];
    for(int i = 0; i < sizeRange; ++i){
        for(int j = 0; j < sizeY; ++j){
            input[i * sizeY + j] = randomF();
        }
    }
    
    dataType *res1 = new dataType[sizeX * sizeY];
    dataType *res2 = new dataType[sizeX * sizeY];
    
    MM_Correct(matrix, input, res1, sizeX, sizeRange, sizeY);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
    gemm(matrix, input, res2, sizeX, sizeRange, sizeY);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaGetLastError());
    double errorRatio = 0;
    for(int i = 0; i < sizeX; ++i){
        for(int j = 0; j < sizeY; ++j){
            if((abs(res1[i * sizeY + j]) + abs(res2[i * sizeY + j])) != 0)
                errorRatio += abs(res1[i * sizeY + j] - res2[i * sizeY + j])/(abs(res1[i * sizeY + j]) + abs(res2[i * sizeY + j]));
        }
    }
    errorRatio /= (sizeX * sizeY);

    delete[] matrix;
    delete[] input;
    delete[] res1;
    delete[] res2;
    
    cout << "Error : " << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << errorRatio << "\n";
    return 0;
}
