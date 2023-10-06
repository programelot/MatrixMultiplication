## Algorithms ##

1. Naive CPU implemented dense-matrix multiplication
2. Naive GPU implemented dense-matrix multiplication
3. Naive GPU implemented dense-matrix multiplication + Registered result 
4. X,Y-coordinated GPU implementated dense-matrix multiplication + Registered result
5. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory
6. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 1
7. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 2
8. X,Y-coordinated GPU implementated blocked dense-matrix multiplication + Registered result + Utilize shared memory + Shared memory indexing method 3
9. Strassen's algorithm
10. Memory usage optimized Strassen's algorithm
11. Winograd's algorithm

#### Experiment setup ####

| GPU | CPU | RAM | OS |
| --- | --- | --- | --- |
| GeForce RTX 2060 | Intel(R) Core(TM) i7-6700 | 16.341768 GB | Ubuntu |

#### Experiment result ####

##### Test case #####
| Index | Matrix size | Input size | Result size | Remark |
| ----- | ----------- | ---------- | ----------- | ------ |
| 1 | 128 x 128 | 128 x 128 | 128 x 128 | Rectangular matrix/input 1 |
| 2 | 1024 x 1024 | 1024 x 1024 | 1024 x 1024 | Rectangular matrix/input 1 |
| 3 | 4096 x 4096 | 4096 x 4096 | 4096 x 4096 | Rectangular matrix/input 2 |
| 4 | 16000 x 16000 | 16000 x 16000 | 16000 x 16000 | Rectangular matrix/input 3 |
| 5 | 16383 x 16383 | 16383 x 16383 | 16383 x 16383 | Rectangular matrix/input 4 (Allways odd size with recursion) |
| 6 | 4096 x 10 | 10 x 4096 | 4096 x 4096 | Small square matrix/input |
| 7 | 100 x 16000 | 16000 x 16000 | 100 x 16000 | Square matrix|
| 8 | 16000 x 100 | 100 x 16000 | 16000 x 16000 | Square matrix/input |
| 9 | 16000 x 16000 | 16000 x 100 | 16000 x 100 | Square input |
| 10 | 100 x 100 | 100 x 32000 | 100 x 32000 | Form 1 |
| 11 | 100 x 32000 | 32000 x 100 | 100 x 100 | Form 2 |

##### Test results #####

##### Execution time #####

| Index | MM1(msec) | MM2(msec) | MM3(msec) | MM4(msec) | MM5(msec) | MM6(msec) |
| ----- | --------- | --------- | --------- | --------- | --------- | --------- |
| 1 | 7.7 | 1.61 | 1.66 | 1.71 | 1.73 | 1.72 | 
| 2 | 9.07K | 26.27 | 18.27 | 67.4 | 74.37 | 68.98 | 
| 3 | 252.31K | 937.43 | 804.42 | 2.55K | 2.83K | 2.76K | 
| 4 | 15.04M | 85.92K | 71.55K | 154.53K | 165.65K | 161.37K | 
| 5 | 16.14M | 97.72K | 98.13K | 85.23K | 171.26K | 166.65K | 
| 6 | 615.99 | 60.36 | 58.82 | 59.91 | 67.84 | 67.78 | 
| 7 | 93.99K | 741.45 | 644.77 | 1.2K | 1.55K | 1.5K | 
| 8 | 93.99K | 877.57 | 782.96 | 1.02K | 1.64K | 1.62K | 
| 9 | 93.99K | 574.45 | 428.97 | 1.3K | 1.64K | 1.61K | 
| 10 | 1.17K | 14.69 | 14.18 | 15.37 | 26.98 | 26.44 | 
| 11 | 1.17K | 108.27 | 100.6 | 126.46 | 136.01 | 141.98 | 

| Index | MM7(msec) | MM8(msec) | MM9(msec) | MM10(msec) | MM11(msec) | MM12(msec) |
| ----- | --------- | --------- | --------- | ---------- | ---------- |---------- | 
| 1 | 1.55 | 1.54 | 1.52 | 1.52 | 1.61 | 1.60 |
| 2 | 22.15 | 20.48 | 18.11 | 18.06 | 18.21 | 18.14 |
| 3 | 492.91 | 422.09 | 320.62 | 315.96 | 305.43 | 345.61 |
| 4 | 24.83K | 20.59K | 12.04K | 11.92K | 11.38K | 1.69K |
| 5 | 21.0K | 16.14K | 13.87K | 13.57K | 12.89K | 18.83K |
| 6 | 59.54 | 60.92 | 57.83 | 58.55 | 60.08 | 59.77 |
| 7 | 430.0 | 392.0 | 633.08 | 621.75 | 596.2 | 596.23 |
| 8 | 768.82 | 739.29 | 781.01 | 780.77 | 772.6 | 765.33 |
| 9 | 513.0 | 484.94 | 522.84 | 509.67 | 478.36 | 454.79 |
| 10 | 12.23 | 11.78 | 14.11 | 14.21 | 14.2 | 14.16 |
| 11 | 103.53 | 102.3 | 101.97 | 102.28 | 102.05 | 102.73 |

##### Throughput #####

| Index | MM1(flops) | MM2(flops) | MM3(flops) | MM4(flops) | MM5(flops) | MM6(flops) |
| ----- | --- | --- | --- | --- | --- | --- |
| 1 | 542.6M | 2.59G | 2.52G | 2.44G | 2.41G | 2.43G | 
| 2 | 236.69M | 81.71G | 117.45G | 31.84G | 28.86G | 31.11G | 
| 3 | 544.66M | 146.59G | 170.83G | 53.82G | 48.54G | 49.8G | 
| 4 | 544.71M | 95.34G | 114.49G | 53.01G | 49.45G | 50.77G | 
| 5 | 544.71M | 90.0G | 89.62G | 103.18G | 51.35G | 52.77G | 
| 6 | 517.49M | 5.28G | 5.42G | 5.32G | 4.7G | 4.7G | 
| 7 | 544.71M | 69.05G | 79.41G | 42.56G | 33.11G | 34.06G | 
| 8 | 542.0M | 58.05G | 65.07G | 50.07G | 31.13G | 31.52G | 
| 9 | 544.71M | 89.13G | 119.35G | 39.39G | 31.26G | 31.84G | 
| 10 | 542.0M | 43.33G | 44.91G | 41.42G | 23.6G | 24.09G | 
| 11 | 544.72M | 5.91G | 6.36G | 5.06G | 4.71G | 4.51G | 

| Index | MM7(flops) | MM8(flops) | MM9(flops) | MM10(flops) | MM11(flops) | MM12(flops) |
| ----- | --- | --- | --- | ---- | ---- | ---- |
| 1 | 2.69G | 2.72G | 2.74G | 2.75G | 2.6G | 2.61G | 
| 2 | 96.92G | 104.78G | 118.49G | 118.87G | 117.9G | 118.35G | 
| 3 | 278.8G | 325.58G | 428.62G | 434.93G | 449.93G | 397.63G | 
| 4 | 329.89G | 397.76G | 680.63G | 687.4G | 720.05G | 485.7G | 
| 5 | 418.69G | 544.95G | 633.98G | 648.3G | 682.43G | 466.95G | 
| 6 | 5.35G | 5.23G | 5.51G | 5.44G | 5.31G | 5.33G | 
| 7 | 119.07G | 130.61G | 80.87G | 82.35G | 85.87G | 85.87G | 
| 8 | 66.26G | 68.91G | 65.23G | 65.25G | 65.94G | 66.56G | 
| 9 | 99.8G | 105.58G | 97.92G | 100.45G | 107.03G | 112.57G | 
| 10 | 52.08G | 54.04G | 45.14G | 44.82G | 44.85G | 44.97G | 
| 11 | 6.18G | 6.26G | 6.28G | 6.26G | 6.27G | 6.23G | 

##### Execution time percentage #####

| Index | MM1(%) | MM2(%) | MM3(%) | MM4(%) | MM5(%) | MM6(%) |
| ----- | --- | --- | --- | --- | --- | --- |
| 1 | 506.78 | 106.02 | 109.26 | 112.62 | 114.11 | 112.94 | 
| 2 | 50.22K | 145.48 | 101.21 | 373.29 | 411.89 | 382.05 | 
| 3 | 82.61K | 306.92 | 263.37 | 835.92 | 926.85 | 903.45 | 
| 4 | 132.19K | 755.23 | 628.92 | 1.36K | 1.46K | 1.42K | 
| 5 | 125.28K | 758.26 | 761.47 | 661.36 | 1.33K | 1.29K | 
| 6 | 1.07K | 104.38 | 101.7 | 103.59 | 117.31 | 117.2 | 
| 7 | 23.98K | 189.15 | 164.48 | 306.91 | 394.52 | 383.52 | 
| 8 | 12.71K | 118.7 | 105.91 | 137.64 | 221.37 | 218.61 | 
| 9 | 21.91K | 133.92 | 100.0 | 303.01 | 381.8 | 374.85 | 
| 10 | 9.97K | 124.71 | 120.33 | 130.46 | 228.97 | 224.36 | 
| 11 | 1.17K | 107.63 | 100.0 | 125.71 | 135.19 | 141.13 | 

| Index | MM7(%) | MM8(%) | MM9(%) | MM10(%) | MM11(%) | MM12(%) |
| ----- | --- | --- | --- | ---- | ---- | ---- |
| 1 | 102.3 | 101.17 | 100.23 | 100.0 | 105.74 | 105.54 | 
| 2 | 122.65 | 113.45 | 100.32 | 100.0 | 100.83 | 100.44 | 
| 3 | 161.38 | 138.19 | 104.97 | 103.45 | 100.0 | 113.15 | 
| 4 | 218.27 | 181.03 | 105.79 | 104.75 | 100.0 | 148.25 | 
| 5 | 162.99 | 125.23 | 107.64 | 105.26 | 100.0 | 146.15 | 
| 6 | 102.95 | 105.34 | 100.0 | 101.24 | 103.88 | 103.34 | 
| 7 | 109.69 | 100.0 | 161.5 | 158.61 | 152.09 | 152.1 | 
| 8 | 104.0 | 100.0 | 105.64 | 105.61 | 104.51 | 103.52 | 
| 9 | 119.59 | 113.05 | 121.88 | 118.81 | 111.52 | 106.02 | 
| 10 | 103.77 | 100.0 | 119.71 | 120.57 | 120.5 | 120.16 | 
| 11 | 102.91 | 101.69 | 101.36 | 101.67 | 101.45 | 102.12 | 

##### Rank #####

| Index | MM1 | MM2 | MM3 | MM4 | MM5 | MM6 | MM7 | MM8 | MM9 | MM10 | MM11 | MM12 |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | ---- | ---- |
| 1 | 12 | 7 | 8 | 9 | 11 | 10 | 4 | 3 | 2 | 1 | 6 | 5 | 
| 2 | 12 | 8 | 5 | 9 | 11 | 10 | 7 | 6 | 2 | 1 | 4 | 3 | 
| 3 | 12 | 8 | 7 | 9 | 11 | 10 | 6 | 5 | 3 | 2 | 1 | 4 | 
| 4 | 12 | 8 | 7 | 9 | 11 | 10 | 6 | 5 | 3 | 2 | 1 | 4 | 
| 5 | 12 | 8 | 9 | 7 | 11 | 10 | 6 | 4 | 3 | 2 | 1 | 5 | 
| 6 | 12 | 8 | 3 | 6 | 11 | 10 | 4 | 9 | 1 | 2 | 7 | 5 | 
| 7 | 12 | 8 | 7 | 9 | 11 | 10 | 2 | 1 | 6 | 5 | 3 | 4 | 
| 8 | 12 | 8 | 7 | 9 | 11 | 10 | 3 | 1 | 6 | 5 | 4 | 2 | 
| 9 | 12 | 8 | 1 | 9 | 11 | 10 | 6 | 4 | 7 | 5 | 3 | 2 | 
| 10 | 12 | 8 | 5 | 9 | 11 | 10 | 2 | 1 | 3 | 7 | 6 | 4 | 
| 11 | 12 | 8 | 1 | 9 | 10 | 11 | 7 | 5 | 2 | 4 | 3 | 6 | 

#### Charts ####
![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Result.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Result_drop_some.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Throughput.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Throughput_drop_some.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Percentage.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Percentage_drop_some.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Testcase5.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Testcase5_drop_some.png?raw=true)

![Experiment result](https://github.com/programelot/MatrixMultiplication/blob/master/DMM/doc/Rank.png?raw=true)
