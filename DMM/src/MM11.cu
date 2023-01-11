#include "GPUdebug.hpp"
#include "common.hpp"

using namespace std;

__global__ void MM_Kernel(dataType* matrix, dataType* input, dataType* res,
        size_t sizeX, size_t sizeRange, size_t sizeY, size_t resUnit, size_t matUnit, size_t inputUnit){
    size_t global = threadIdx.x + blockIdx.x * blockDim.x;
    size_t i = global/sizeY;
    size_t j = global%sizeY;
    global = i * resUnit + j;
    if(i >= sizeX || j >= sizeY) return;
    dataType acc = 0;
    for(int k = 0; k < sizeRange; ++k){
        acc +=  matrix[i * matUnit + k] * input[k * inputUnit + j];
    }
    res[global] = acc;
}

__global__ void SetPlusM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] = from[i * fromUnit + j];
}

__global__ void SetMinusM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] = -from[i * fromUnit + j];
}

__global__ void SetZero_Kernel(dataType *mat, size_t Unit, size_t length){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= length) return;
    mat[i * Unit] = 0;
}

__global__ void AddM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] += from[i * fromUnit + j];
}

__global__ void SubM_Kernel(dataType *from, size_t fromUnit,
            dataType *to,   size_t toUnit,
            size_t height, size_t width){
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= height || j >= width) return;
    to[i * toUnit + j] -= from[i * fromUnit + j];
}

bool isBaseCase(size_t sizeX, size_t sizeRange, size_t sizeY){
    return sizeX * sizeRange<= 1048576 || sizeRange * sizeY <= 1048576 || sizeX * sizeY <= 1048576 ||
        sizeX <= 32 || sizeRange <= 32 || sizeY <= 32;
    //return sizeX <= 1024 || sizeRange <= 1024 || sizeY <= 1024;
}

//Copy array from "from" to "to"
//Index will starts from "fromOffset" and "toOffset"
//It will copy width in a row for height times with each corresponding units.
void addMat(dataType *from, size_t fromUnit,
            dataType *to, size_t toUnit,
            size_t height, size_t width, bool plus){
    dim3 threadDim(height > subBlockSize ? subBlockSize : height , width > subBlockSize ? subBlockSize : width);
    dim3 blockDim((height + subBlockSize - 1)/subBlockSize , (width + subBlockSize - 1)/subBlockSize);
    if(plus){
        AddM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
    }
    else{
        SubM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
    }
}

//Set matrix value <from> to <to>
//Fill zero for one line of row or col or both if it needed.
void setMat(dataType *from, size_t fromUnit,
            dataType *to, size_t toUnit,
            size_t height, size_t width, 
            bool extendHeight, bool extendWidth, bool plus){
    dim3 threadDim(height > subBlockSize ? subBlockSize : height , width > subBlockSize ? subBlockSize : width);
    dim3 blockDim((height + subBlockSize - 1)/subBlockSize , (width + subBlockSize - 1)/subBlockSize);

    
    if(plus){
        SetPlusM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
    }
    else{
        SetMinusM_Kernel<<<blockDim, threadDim>>>(from, fromUnit, to, toUnit, height, width);
    }

    if(extendHeight){
        // SetZero_Kernel<<<((width + (extendWidth ? 1 : 0)) + MAXTHREADSIZE - 1) / MAXTHREADSIZE,
        //                   (width + (extendWidth ? 1 : 0)) > MAXTHREADSIZE? MAXTHREADSIZE : (width + (extendWidth ? 1 : 0))>>>
        //                   (to + (height * toUnit), 1, (width + (extendWidth ? 1 : 0)));
        HANDLE_ERROR(cudaMemset(to + (height * toUnit), 0,  sizeof(dataType) * (width + (extendWidth ? 1 : 0))));
    }

    if(extendWidth){
        SetZero_Kernel<<<(height + MAXTHREADSIZE - 1) / MAXTHREADSIZE, height > MAXTHREADSIZE? MAXTHREADSIZE : height>>>(to + width, toUnit, height);
    }
}

void gemm(  dataType * matrix, dataType * input, dataType * result,
            size_t matUnit, size_t inpUnit, size_t resUnit,
            dataType * auxMat, dataType * auxInp, dataType * auxRes,
            size_t auxMatUnit, size_t auxInpUnit, size_t auxResUnit,
            size_t sizeX, size_t sizeRange, size_t sizeY ){

    bool baseCase = isBaseCase(sizeX, sizeRange, sizeY);
    if(baseCase){
        size_t work = sizeX * sizeY;
        MM_Kernel<<<(work + MAXTHREADSIZE - 1)/MAXTHREADSIZE, work > MAXTHREADSIZE? MAXTHREADSIZE : work >>>(matrix, input, result, sizeX, sizeRange, sizeY, resUnit, matUnit, inpUnit);
    }
    else{
        const size_t rowNumMat = sizeX;
        const size_t colNumMat = sizeRange;
        const size_t rowNumInp = sizeRange;
        const size_t colNumInp = sizeY;
        const size_t rowNumRes = sizeX;
        const size_t colNumRes = sizeY;
        
        const size_t rowNumAuxmat = (rowNumMat + 1)/2;
        const size_t colNumAuxmat = (colNumMat + 1)/2;
        const size_t rowNumAuxinp = (rowNumInp + 1)/2;
        const size_t colNumAuxinp = (colNumInp + 1)/2;
        const size_t rowNumAuxres = (rowNumRes + 1)/2;
        const size_t colNumAuxres = (colNumRes + 1)/2;

        const size_t rowNumAuxmatLeft = rowNumMat - rowNumAuxmat;
        const size_t colNumAuxmatLeft = colNumMat - colNumAuxmat;
        const size_t rowNumAuxinpLeft = rowNumInp - rowNumAuxinp;
        const size_t colNumAuxinpLeft = colNumInp - colNumAuxinp;
        const size_t rowNumAuxresLeft = rowNumRes - rowNumAuxres;
        const size_t colNumAuxresLeft = colNumRes - colNumAuxres;
        
        const size_t auxMatSize = rowNumAuxmat * colNumAuxmat;
        const size_t auxInpSize = rowNumAuxinp * colNumAuxinp;
        const size_t auxResSize = rowNumAuxres * colNumAuxres;

        const size_t auxSizeX     = rowNumAuxres;
        const size_t auxSizeRange = colNumAuxmat;
        const size_t auxSizeY     = colNumAuxres;
        
        const size_t auxSizeXLeft     = rowNumAuxresLeft;
        const size_t auxSizeRangeLeft = colNumAuxmatLeft;
        const size_t auxSizeYLeft     = colNumAuxresLeft;
        
        const size_t rowNumAuxauxmat = (rowNumAuxmat + 1)/2;
        const size_t colNumAuxauxmat = (colNumAuxmat + 1)/2;
        const size_t rowNumAuxauxinp = (rowNumAuxinp + 1)/2;
        const size_t colNumAuxauxinp = (colNumAuxinp + 1)/2;
        const size_t rowNumAuxauxres = (rowNumAuxres + 1)/2;
        const size_t colNumAuxauxres = (colNumAuxres + 1)/2;

        dataType* auxAuxMat = auxMat + auxMatSize;
        dataType* auxAuxInp = auxInp + auxInpSize;
        dataType* auxAuxRes = auxRes + auxResSize;

        dataType* mat_11 = matrix;
        dataType* mat_12 = mat_11 + colNumAuxmat;
        dataType* mat_21 = mat_11 + matUnit * rowNumAuxmat;
        dataType* mat_22 = mat_21 + colNumAuxmat;

        dataType* inp_11 = input;
        dataType* inp_12 = inp_11 + colNumAuxinp;
        dataType* inp_21 = inp_11 + inpUnit * rowNumAuxinp;
        dataType* inp_22 = inp_21 + colNumAuxinp;

        dataType* res_11 = result;
        dataType* res_12 = res_11 + colNumAuxres;
        dataType* res_21 = res_11 + resUnit * rowNumAuxres;
        dataType* res_22 = res_21 + colNumAuxres;
        
        ////////////////////////////////////////////////
        // S3 applied
        // S3 = mat_11 - mat_21
        ////////////////////////////////////////////////

        setMat( mat_11, matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmat, colNumAuxmat, 
                false, false, true);
                
        addMat( mat_21, matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmatLeft, colNumAuxmat, false);
        
        ////////////////////////////////////////////////
        // T3 applied
        // T3 =  - inp_12 + inp_22
        ////////////////////////////////////////////////

        setMat( inp_12,  inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinp, colNumAuxinpLeft, 
                false, colNumAuxinpLeft != colNumAuxinp, false);

        addMat( inp_22,  inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinpLeft, colNumAuxinpLeft, true);
        
        ////////////////////////////////////////////////
        // M7 applied
        // M7 = S3 T3
        // positive: res_21
        // negative: None
        ////////////////////////////////////////////////
        
        gemm( auxMat, auxInp, auxRes,
              auxMatUnit, auxInpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRange, auxSizeY);
              
        setMat( auxRes,  auxResUnit, 
                res_21, resUnit,
                rowNumAuxresLeft, colNumAuxres, false, false, true);

        ////////////////////////////////////////////////
        // M1 applied
        // M1 = mat_11 inp_11
        // positive: res_11
        // negative: None
        ////////////////////////////////////////////////
        
        gemm( mat_11, inp_11, auxRes,
              matUnit, inpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRange, auxSizeY);
              
        setMat( auxRes,  auxResUnit, 
                res_11, resUnit,
                rowNumAuxres, colNumAuxres, false, false, true);

        ////////////////////////////////////////////////
        // S1 applied
        // S1 = mat_21 + mat_22
        ////////////////////////////////////////////////

        setMat( mat_21, matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmatLeft, colNumAuxmat, 
                rowNumAuxmatLeft != rowNumAuxmat, false, true);
        
        addMat( mat_22, matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmatLeft, colNumAuxmatLeft, true);

        ////////////////////////////////////////////////
        // T1 applied
        // T1 =  - inp_11 + inp_12
        ////////////////////////////////////////////////
        
        setMat( inp_11,  inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinp, colNumAuxinp, false, false, false);

        addMat( inp_12,  inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinp, colNumAuxinpLeft, true);


        ////////////////////////////////////////////////
        // M5 applied
        // M5 = S1T1
        // positive: res_12, res_22
        // negative: None
        ////////////////////////////////////////////////

        gemm( auxMat, auxInp, auxRes,
              auxMatUnit, auxInpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRange, auxSizeY);
              
        setMat( auxRes,  auxResUnit, 
                res_12, resUnit,
                rowNumAuxres, colNumAuxresLeft, false, false, true);
                
        setMat( auxRes,  auxResUnit, 
                res_22, resUnit,
                rowNumAuxresLeft, colNumAuxresLeft, false, false, true);

        ////////////////////////////////////////////////
        // S2 applied
        // S2 = S1 - mat_11
        ////////////////////////////////////////////////
        
        addMat( mat_11, matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmat, colNumAuxmat, false);


        ////////////////////////////////////////////////
        // -T2 applied
        // -T2 = T1 - inp_22
        ////////////////////////////////////////////////

        addMat( inp_22,  inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinpLeft, colNumAuxinpLeft, false);

        ////////////////////////////////////////////////
        // -M6 applied
        // -M6 = S2 * -T2
        // positive: None
        // negative: None
        ////////////////////////////////////////////////

        gemm( auxMat, auxInp, auxRes,
              auxMatUnit, auxInpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRange, auxSizeY);

        ////////////////////////////////////////////////
        // -U2 applied
        // -U2 = -M6 - M1(from res_11)
        ////////////////////////////////////////////////

        addMat( res_11, resUnit, 
                auxRes, auxResUnit,
                rowNumAuxres, colNumAuxres, false);

        ////////////////////////////////////////////////
        // U4 applied
        // U4(at res_12) = M5 -(-U2)
        ////////////////////////////////////////////////

        addMat( auxRes, auxResUnit, 
                res_12, resUnit,
                rowNumAuxres, colNumAuxresLeft, false);

        ////////////////////////////////////////////////
        // U3 applied
        // U3(at res_21) = M7(at res_21) -(-U2)
        ////////////////////////////////////////////////

        addMat( auxRes, auxResUnit, 
                res_21, resUnit,
                rowNumAuxresLeft, colNumAuxres, false);

        ////////////////////////////////////////////////
        // U7 applied
        // U7(at res_22) = U3(at res_21) + M5(at res_22)
        ////////////////////////////////////////////////

        addMat( res_21, resUnit, 
                res_22, resUnit,
                rowNumAuxresLeft, colNumAuxresLeft, true);

        ////////////////////////////////////////////////
        // -S4 applied
        // -S4 = S2 - mat_12
        ////////////////////////////////////////////////

        addMat( mat_12,  matUnit, 
                auxMat, auxMatUnit,
                rowNumAuxmat, colNumAuxmatLeft, false);

        ////////////////////////////////////////////////
        // -T4 applied
        // -T4 =  inp_21 + (-T2)
        ////////////////////////////////////////////////

        addMat( inp_21, inpUnit, 
                auxInp, auxInpUnit,
                rowNumAuxinpLeft, colNumAuxinp, true);

        ////////////////////////////////////////////////
        // -M3 applied
        // -M3 = (-S4) inp_22
        ////////////////////////////////////////////////

        gemm( auxMat, inp_22, auxRes,
              auxMatUnit, inpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRangeLeft, auxSizeYLeft);

        ////////////////////////////////////////////////
        // U5 applied
        // U5(at res_12) = U4(at res_12) - (-M3)
        ////////////////////////////////////////////////
        
        addMat( auxRes, auxResUnit, 
                res_12, resUnit,
                rowNumAuxres, colNumAuxresLeft, false);

        ////////////////////////////////////////////////
        // -M4 applied
        // -M4 = mat_22 (-T4)
        ////////////////////////////////////////////////

        gemm( mat_22, auxInp, auxRes,
              matUnit, auxInpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeXLeft, auxSizeRangeLeft, auxSizeY);

        ////////////////////////////////////////////////
        // U6 applied
        // U6(at res_21) = U3(at res_21) + (-M4)
        ////////////////////////////////////////////////

        addMat( auxRes, auxResUnit, 
                res_21, resUnit,
                rowNumAuxresLeft, colNumAuxres, true);

        ////////////////////////////////////////////////
        // M2 applied
        // M2 = mat_12 inp_21
        ////////////////////////////////////////////////

        gemm( mat_12, inp_21, auxRes,
              matUnit, inpUnit, auxResUnit,
              auxAuxMat, auxAuxInp, auxAuxRes, 
              colNumAuxauxmat, colNumAuxauxinp, colNumAuxauxres,
              auxSizeX, auxSizeRangeLeft, auxSizeY);

        ////////////////////////////////////////////////
        // U1 applied
        // U1(res_11) = M1 + M2
        ////////////////////////////////////////////////

        addMat( auxRes, auxResUnit, 
                res_11, resUnit,
                rowNumAuxres, colNumAuxres, true);
    }
}


float gemm(dataType* matrix, dataType* input, dataType* res,
    const size_t rowSize, const size_t rangeSize, const size_t colSize){
    dataType *matrix_GPU, *input_GPU, *res_GPU;
    
    const size_t rowNumMat = rowSize;
    const size_t colNumMat = rangeSize;
    const size_t rowNumInp = rangeSize;
    const size_t colNumInp = colSize;
    const size_t rowNumRes = rowSize;
    const size_t colNumRes = colSize;

    //Get auxiliary requirement size
    size_t auxSize_Mat = 0;
    size_t auxSize_Inp = 0;
    size_t auxSize_Res = 0;
    {
        size_t auxRow = rowSize;
        size_t auxRange = rangeSize;
        size_t auxCol = colSize;
        while(!isBaseCase(auxRow, auxRange, auxCol)){//Minimum matrix multiplication is 1024
            auxRow = (auxRow + 1)/2;
            auxCol = (auxCol + 1)/2;
            auxRange = (auxRange + 1)/2;
            
            auxSize_Mat += auxRow * auxRange;
            auxSize_Inp += auxRange * auxCol;
            auxSize_Res += auxRow * auxCol;
        }
    }

    HANDLE_ERROR(cudaMalloc(&matrix_GPU, sizeof(dataType)*(rowNumMat*colNumMat + auxSize_Mat)));
    HANDLE_ERROR(cudaMalloc(&input_GPU,  sizeof(dataType)*(rowNumInp*colNumInp + auxSize_Inp)));
    HANDLE_ERROR(cudaMalloc(&res_GPU,    sizeof(dataType)*(rowNumRes*colNumRes + auxSize_Res)));
    HANDLE_ERROR(cudaMemcpy(matrix_GPU, matrix, sizeof(dataType) * colNumMat * rowNumMat, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(input_GPU, input, sizeof(dataType) * colNumInp * rowNumInp, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));

    const size_t matSize = colNumMat * rowNumMat;
    const size_t inpSize = colNumInp * rowNumInp;
    const size_t resSize = colNumRes * rowNumRes;

    gemm(   matrix_GPU, input_GPU, res_GPU,
            colNumMat, colNumInp, colNumRes,
            matrix_GPU + matSize, input_GPU + inpSize, res_GPU + resSize,
            (colNumMat + 1)/2, (colNumInp + 1)/2, (colNumRes + 1)/2,
            rowNumMat, colNumMat, colNumInp);
    HANDLE_ERROR(cudaMemcpy(res, res_GPU, sizeof(dataType) * colNumRes * rowNumRes, cudaMemcpyDeviceToHost));
    
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    HANDLE_ERROR(cudaFree(matrix_GPU));
    HANDLE_ERROR(cudaFree(input_GPU));
    HANDLE_ERROR(cudaFree(res_GPU));
    return milliseconds;
}