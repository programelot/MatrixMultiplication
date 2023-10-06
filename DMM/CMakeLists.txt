cmake_minimum_required(VERSION 3.19 FATAL_ERROR) #Specify the minimum cmake version to use
project(denseMatrixMultiplication LANGUAGES CXX CUDA) #Project name and languages in use

find_package(CUDAToolkit) #find cuda library and compiler

include_directories(include)

add_library(MML1  src/MM1.cu) # cpp
add_library(MML2  src/MM2.cu) # res major
add_library(MML3  src/MM3.cu) # res major + local output
add_library(MML4  src/MM4.cu) # res major + two axis + local output
add_library(MML5  src/MM5.cu) # res major + blocked + shared + local output
add_library(MML6  src/MM6.cu) # res major + blocked + shared + local output + coealocal 1
add_library(MML7  src/MM7.cu) # res major + blocked + shared + local output + coealocal 2
add_library(MML8  src/MM8.cu) # res major + blocked + shared + local output + coealocal 3
add_library(MML9  src/MM9.cu) # Strassen's
add_library(MML10 src/MM10.cu) # Data transfer optimized Strassen's
add_library(MML11 src/MM11.cu) # Winograd's

add_executable(MMP1  src/tests/performance.cu)
target_link_libraries(MMP1 MML1)
add_executable(MMP2  src/tests/performance.cu)
target_link_libraries(MMP2 MML2)
add_executable(MMP3  src/tests/performance.cu)
target_link_libraries(MMP3 MML3)
add_executable(MMP4  src/tests/performance.cu)
target_link_libraries(MMP4 MML4)
add_executable(MMP5  src/tests/performance.cu)
target_link_libraries(MMP5 MML5)
add_executable(MMP6  src/tests/performance.cu)
target_link_libraries(MMP6 MML6)
add_executable(MMP7  src/tests/performance.cu)
target_link_libraries(MMP7 MML7)
add_executable(MMP8  src/tests/performance.cu)
target_link_libraries(MMP8 MML8)
add_executable(MMP9  src/tests/performance.cu)
target_link_libraries(MMP9 MML9)
add_executable(MMP10 src/tests/performance.cu)
target_link_libraries(MMP10 MML10)
add_executable(MMP11 src/tests/performance.cu)
target_link_libraries(MMP11 MML11)

add_executable(MMEP1  src/tests/errorGPU.cu)
target_link_libraries(MMEP1 MML1)
add_executable(MMEP2  src/tests/errorGPU.cu)
target_link_libraries(MMEP2 MML2)
add_executable(MMEP3  src/tests/errorGPU.cu)
target_link_libraries(MMEP3 MML3)
add_executable(MMEP4  src/tests/errorGPU.cu)
target_link_libraries(MMEP4 MML4)
add_executable(MMEP5  src/tests/errorGPU.cu)
target_link_libraries(MMEP5 MML5)
add_executable(MMEP6  src/tests/errorGPU.cu)
target_link_libraries(MMEP6 MML6)
add_executable(MMEP7  src/tests/errorGPU.cu)
target_link_libraries(MMEP7 MML7)
add_executable(MMEP8  src/tests/errorGPU.cu)
target_link_libraries(MMEP8 MML8)
add_executable(MMEP9  src/tests/errorGPU.cu)
target_link_libraries(MMEP9 MML9)
add_executable(MMEP10 src/tests/errorGPU.cu)
target_link_libraries(MMEP10 MML10)
add_executable(MMEP11 src/tests/errorGPU.cu)
target_link_libraries(MMEP11 MML11)