#include <iostream>
#include <time.h>
#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;
extern float gemm(dataType* matrix, dataType* input, dataType* res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize);

dataType randomF(){
    return (rand()%3) - 1;
}

int main(int argc, char **argv){
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    srand(time(NULL));

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
    
    dataType *res = new dataType[sizeX * sizeY];
    HANDLE_ERROR(cudaEventRecord(start));
    float kernel_time = gemm(matrix, input, res, sizeX, sizeRange, sizeY);
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    cout << milliseconds << " " << kernel_time << "\n";

    delete[] matrix;
    delete[] input;
    delete[] res;
    
    return 0;
}
