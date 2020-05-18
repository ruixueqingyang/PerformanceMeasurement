
/*******************************************************************************
Copyright(C), 2020-2020, 瑞雪轻飏
     FileName: PowerManager.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20200514
  Description: 能耗控制
       Others: 
*******************************************************************************/

#include "PowerManager.h"

// 可用 F-F 配置对数量
#define K40M "Tesla K40m"
const int K40mClkNum = 5;
// 可用 F-F 配置对
const struct GPU_CLK K40mClk[5] = {{3004, 875}, {3004, 810}, {3004, 745}, {3004, 666}, {324, 324}};
// 最大功率 W
const int K40mMaxPower = 235;

int POWER_MANAGER::initArg(){

    if(TuneType < 0){
        std::cout << "Power Manager: Use default power strategy" << std::endl;
    }

    nvmlResult = nvmlInit();
    if (NVML_SUCCESS != nvmlResult)
    {
        std::cout << "Power Manager ERROR: Failed to get device handle (NVML ERROR INFO: " << nvmlErrorString(nvmlResult) << ")." << std::endl;
        exit(-1);
    }
    isNVMLInit = true;

    nvmlResult = nvmlDeviceGetHandleByIndex(indexGPU, &device);
    if(NVML_SUCCESS != nvmlResult){
        std::cout << "Power Manager ERROR: Failed to get device handle (NVML ERROR INFO: " << nvmlErrorString(nvmlResult) << "). Invalid GPU index: -i " << optarg << "." << std::endl;
        exit(-1);
    }

    cudaError_t CUDAErr;
    cudaDeviceProp deviceProp;
    CUDAErr = cudaGetDeviceProperties(&deviceProp, indexGPU);
    if(CUDAErr != cudaSuccess){
        std::cout << "cudaGetDeviceCount ERROR: " << cudaGetErrorString(CUDAErr) << std::endl;
        exit(-1);
    }
    GPUName = deviceProp.name;

    const int* pGPUClkNum;
    const int* pGPUMaxPower;
    const struct GPU_CLK* pGPUClk;

    if(GPUName == K40M){
        pGPUClkNum = &K40mClkNum;
        pGPUMaxPower = &K40mMaxPower;
        pGPUClk = K40mClk;
    }else{
        std::cout << "Power Manager ERROR: Invalid GPU type (GPUName = " << GPUName << ")." << std::endl;
        exit(-1);
    }

    if(TuneType == 0){
        if(0<=TuneArg && TuneArg<*pGPUClkNum){
            indexClockPair = TuneArg;
            memClockMHz = pGPUClk[indexClockPair].MemClk;
            graphicsClockMHz = pGPUClk[indexClockPair].SMClk;
        }else{
            std::cout << "Power Manager ERROR: Invalid F-F pair index (TuneArg = " << TuneArg << ")." << std::endl;
            exit(-1);
        }
    }else if(TuneType == 1){
        if(TuneArg<=0 || TuneArg>*pGPUMaxPower){
            std::cout << "Power Manager ERROR: Invalid power limit (TuneArg = " << TuneArg << ")." << std::endl;
            exit(-1);
        }else{
            powerLimit = TuneArg;
        }
    }else if(TuneType > 1){
        std::cout << "Power Manager ERROR: Invalid tune type (TuneType = " << TuneType << ")" << std::endl;
        exit(-1);
    }

    return 0;
}

int POWER_MANAGER::initCLI(int argc, char** argv){

    init();

    int err = 0;
    int TmpValue = -1;
    extern char *optarg;
    extern int optind, opterr, optopt;
    int c;
    const char *optstring = "i:t:p:";
    // -i indexGPU
    // -t 调节技术类型
    // -p 配置参数: F-F 对的 index 或 功率上限
    while ((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'i':
                indexGPU = atoi(optarg);
                if(indexGPU<0){
                    std::cout << "Power Manager ERROR: Invalid GPU index (-i " << optarg << ")." << std::endl;
                    err = -1;
                }
                break;
            case 't':
                if(strcmp("DVFS", optarg)==0){
                    TuneType = 0;
                }else if(strcmp("POWER", optarg)==0){
                    TuneType = 1;
                }else{
                    std::cout << "Power Manager ERROR: Invalid tuning technology type (-t " << optarg << ")." << std::endl;
                    err = -1;
                }
                break;
            case 'p':
                TmpValue = atoi(optarg);
                if(TmpValue<0){
                    std::cout << "Power Manager ERROR: Invalid configuration parameter (-p " << optarg << ")." << std::endl;
                    err = -1;
                }
                break;
            case '?':
                printf("error optopt: %c\n", optopt);
                printf("error opterr: %d\n", opterr);
                err = -1;
                break;
        }
    }

    if(err!= 0) exit(err);
    err = initArg();
    if(err!= 0) exit(err);

    return 0;
}

int POWER_MANAGER::Set(){
    if(TuneType==0){
        std::cout << "DVFS: memClockMHz = " << memClockMHz << " MHz, graphicsClockMHz = " << graphicsClockMHz << " MHz" << std::endl;

        nvmlResult = nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz);

        if(NVML_SUCCESS != nvmlResult){
            printf("Power Manager ERROR: Failed to DVFS (NVML ERROR INFO: %s)\n", nvmlErrorString(nvmlResult));
            exit(-1);
        }
    }else if(TuneType==1){
        std::cout << "PowerCap: powerLimit = " << powerLimit << " W" << std::endl;

        nvmlResult = nvmlDeviceSetPowerManagementLimit(device, powerLimit*1000);

        if(NVML_SUCCESS != nvmlResult){
            printf("Power Manager ERROR: Failed to set power cap (NVML ERROR INFO: %s)\n", nvmlErrorString(nvmlResult));
            exit(-1);
        }
    }
    return 0;
}

int POWER_MANAGER::Reset(){
    if(TuneType==0){

        nvmlResult = nvmlDeviceResetApplicationsClocks(device);

        if(NVML_SUCCESS != nvmlResult){
            printf("Power Manager ERROR: Failed to reset DVFS (NVML ERROR INFO: %s)\n", nvmlErrorString(nvmlResult));
            exit(-1);
        }
    }else if(TuneType==1){

        unsigned int defaultLimit;

        nvmlResult = nvmlDeviceGetPowerManagementDefaultLimit(device, &defaultLimit);
        if(NVML_SUCCESS != nvmlResult){
            printf("Power Manager ERROR: Failed to get default power cap (NVML ERROR INFO: %s)\n", nvmlErrorString(nvmlResult));
            exit(-1);
        }


        nvmlResult = nvmlDeviceSetPowerManagementLimit(device, defaultLimit);

        if(NVML_SUCCESS != nvmlResult){
            printf("Power Manager ERROR: Failed to reset power cap (NVML ERROR INFO: %s)\n", nvmlErrorString(nvmlResult));
            exit(-1);
        }
    }

    return 0;
}