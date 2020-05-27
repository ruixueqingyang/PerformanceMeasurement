/*******************************************************************************
Copyright(C), 2020-2020, 瑞雪轻飏
     FileName: PowerManager.h
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20200506
  Description: 1. 包含各种头文件
               2. 定义 可用的频率对
       Others: //其他内容说明
*******************************************************************************/

#ifndef __POWER_MANAGER_H
#define __POWER_MANAGER_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <system_error>
#include <string.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <math.h>
#include <fstream>

#include <pthread.h>
#include <unistd.h>
#include <getopt.h>
#include <nvml.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <semaphore.h>

#include <cuda.h>
#include <cuda_runtime.h>

struct GPU_CLK{
    unsigned int MemClk;
    unsigned int SMClk;
};

class POWER_MANAGER{
public:
    std::string GPUName; // -g GPU 型号
    int TuneType; // -t 调节类型: "DVFS" 0 DVFS; "POWER" 1 功率上限调节
    int indexGPU; // -i GPU 序号
    int indexClockPair; // -p F-F 对的 index
    unsigned int memClockMHz;
    unsigned int graphicsClockMHz;
    unsigned int powerLimit; // -p W
    nvmlReturn_t nvmlResult;
    nvmlDevice_t device;

    int TuneArg; // 临时保存调节参数, 初始化时使用

    bool isNVMLInit;

    const int* pGPUClkNum;
    const int* pGPUMaxPower;
    const struct GPU_CLK* pGPUClk;

    int init(){
        TuneType = -1;
        indexGPU = 0;
        indexClockPair = -1;
        powerLimit = 0;
        TuneArg = -1;
        isNVMLInit = false;

        pGPUClkNum = NULL;
        pGPUMaxPower = NULL;
        pGPUClk = NULL;

        return 0;
    }

    int initArg();
    int initArg(int inIndexGPU, int inTuneType, int inTuneArg);
    int initCLI(int argc, char** argv);
    
    POWER_MANAGER(int argc, char** argv){
        initCLI(argc, argv);
    }
    POWER_MANAGER(){
        init();
    }
    ~POWER_MANAGER(){
    }

    int Set();
    int Set(int inTuneArg);
    int Reset();
};


#endif