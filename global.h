/*******************************************************************************
Copyright(C), 2020-2020, 瑞雪轻飏
     FileName: global.h
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20200506
  Description: 1. 包含各种头文件
               2. 定义 配置信息类(CONFIG) 以及 测量数据类(PERF_DATA)
       Others: //其他内容说明
*******************************************************************************/

#ifndef __GLOBAL_H
#define __GLOBAL_H

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

#include <cuda.h>
#include <cuda_runtime.h>

#define VECTOR_RESERVE 20000

enum MEASURE_MODEL {
    INTERACTION,
    DURATION,
    APPLICATION
};

class CONFIG {
public:

    MEASURE_MODEL MeasureModel;

    bool isGenOutFile, isMeasureEnergy, isMeasureMemUtil, isMeasureMemClk, isMeasureGPUUtil, isMeasureSMClk;

    int DeviceID;
    std::string OutFilePath;

    float SampleInterval; // (ms) Sampling Interval
    float PowerThreshold; // (W) Part of power above this threshold is consider as dynamic power consumed by applications
    float MeasureDuration; // (s) Sampling will keep for the specific measurement duration
    float PostInterval; // (s) Sampling will keep for the specific time interval after all applications are completed

    std::vector< std::vector< std::string > > AppPathes;

    void init();
    void init(std::string ConfigDir);

    CONFIG(){
        init();
    }
    CONFIG(std::string ConfigDir) {
        init(ConfigDir);
    }
    //~CONFIG();
};

void CONFIG::init(){

    MeasureModel = MEASURE_MODEL::INTERACTION;

    isGenOutFile = false;
    isMeasureEnergy = true;
    isMeasureMemUtil = true;
    isMeasureMemClk = true;
    isMeasureGPUUtil = true;
    isMeasureSMClk = true;

    DeviceID = 0;

    SampleInterval = 20.0;
    PowerThreshold = 20.0;
    MeasureDuration = -1.0;
    PostInterval = 0.0;
}

void CONFIG::init(std::string ConfigDir){
    init();
    // 下边读取文件, 设定参数, 以后再写
}

class PERF_DATA{
public:
    CUdevice cuDevice;
    nvmlDevice_t nvmlDevice;
    int ComputeCapablityMajor;
    int ComputeCapablityMinor;
    unsigned int long long SampleCount;
    double TotalDuration; // (s)

    struct timeval prevTimeStamp, currTimeStamp;
    std::vector<double> vecTimeStamp; // (s)
    
    unsigned int prevPower, currPower; // (mW)
    nvmlUtilization_t prevUtil, currUtil; // (%)
    unsigned int prevMemClk, currMemClk, prevSMClk, currSMClk; // (MHz)
    // float prevMemUtil, currGPUUtil;
    std::vector<unsigned int> vecPower, vecMemUtil, vecMemClk, vecGPUUtil, vecSMClk; // (mW, %, MHz, %, MHz)

    bool isFisrtSample;

    unsigned int minPower, maxPower; // (mW, mW)
    float avgPower, Energy; // (mW, mJ)

    unsigned int minMemUtil, maxMemUtil; // (%, %)
    float avgMemUtil, sumMemUtil; // (%, %)

    unsigned int minMemClk, maxMemClk; // (MHz, MHz)
    float avgMemClk, sumMemClk; // (MHz, MHz)

    unsigned int minGPUUtil, maxGPUUtil; // (%, %)
    float avgGPUUtil, sumGPUUtil; // (%, %)

    unsigned int minSMClk, maxSMClk; // (MHz, MHz)
    float avgSMClk, sumSMClk; // (MHz, MHz)

    int init(){
        SampleCount = 0;
        TotalDuration = 0.0;
        vecTimeStamp.push_back(0.0);
        minPower = 0xFFFFFFFF; maxPower = 0; avgPower = 0.0; Energy = 0.0;
        minMemUtil = 0xFFFFFFFF; maxMemUtil = 0; avgMemUtil = 0.0; sumMemUtil = 0.0;
        minMemClk = 0xFFFFFFFF; maxMemClk = 0; avgMemClk = 0.0; sumMemClk = 0.0;
        minGPUUtil = 0xFFFFFFFF; maxGPUUtil = 0; avgGPUUtil = 0.0; sumGPUUtil = 0.0;
        minSMClk = 0xFFFFFFFF; maxSMClk = 0; avgSMClk = 0.0; sumSMClk = 0.0;
        prevTimeStamp.tv_sec = 0;
        prevTimeStamp.tv_usec = 0;
        isFisrtSample = true;

        return 0;
    }

    int init(CONFIG& Config){
    
        init();

        if(Config.isGenOutFile==false) return 0;

        if( Config.MeasureModel==MEASURE_MODEL::APPLICATION
            || Config.MeasureModel==MEASURE_MODEL::INTERACTION ){
            vecTimeStamp.reserve(VECTOR_RESERVE);
            vecPower.reserve(VECTOR_RESERVE);
            vecMemUtil.reserve(VECTOR_RESERVE);
            vecMemClk.reserve(VECTOR_RESERVE);
            vecGPUUtil.reserve(VECTOR_RESERVE);
            vecSMClk.reserve(VECTOR_RESERVE);
        }else{
            unsigned long long vecLength = Config.MeasureDuration * 1000 / Config.SampleInterval + 1;
            vecTimeStamp.reserve(vecLength);
            vecPower.reserve(vecLength);
            vecMemUtil.reserve(vecLength);
            vecMemClk.reserve(vecLength);
            vecGPUUtil.reserve(vecLength);
            vecSMClk.reserve(vecLength);
        }

        return 0;
    }

    PERF_DATA(){
        init();
    }

    PERF_DATA(CONFIG& Config){
        init(Config);
    }

    int output(CONFIG& Config){

        std::cout << std::endl;
        std::cout << "-------- Performance Measurement Results --------" << std::endl;
        std::cout << "Actual Measurement Duration: " << TotalDuration << " s" << std::endl;
        std::cout << "Actual Sampling Count: " << SampleCount << std::endl;

        if(Config.isMeasureEnergy==true){
            avgPower = Energy / TotalDuration;
            std::cout << std::endl;
            std::cout << "-------- Energy&Power (All) --------" << std::endl;
            std::cout << "Energy: " << Energy/1000 << " J" << std::endl;
            std::cout << "minPower: " << ((float)minPower)/1000 << " W; avgPower: " << avgPower/1000 << " W; maxPower: " << ((float)maxPower)/1000 << " W" << std::endl;
        }

        if(Config.isMeasureEnergy==true){
            avgPower = Energy / TotalDuration;
            float EnergyAbove = Energy/1000 - Config.PowerThreshold*TotalDuration;
            std::cout << std::endl;
            std::cout << "-------- Energy&Power (Above Threshold) --------" << std::endl;
            std::cout << "Energy: " << EnergyAbove << " J" << std::endl;
            std::cout << "minPower: " << ((float)minPower)/1000-Config.PowerThreshold << " W; avgPower: " << avgPower/1000-Config.PowerThreshold << " W; maxPower: " << ((float)maxPower)/1000-Config.PowerThreshold << " W" << std::endl;
        }

        if(Config.isMeasureGPUUtil==true || Config.isMeasureSMClk==true){
            std::cout << std::endl;
            std::cout << "-------- GPU SM --------" << std::endl;
        }
        if(Config.isMeasureGPUUtil==true){
            avgGPUUtil = sumGPUUtil / TotalDuration;
            std::cout << "minGPUUtil: " << minGPUUtil << " %; avgGPUUtil: " << avgGPUUtil << " %; maxGPUUtil: " << maxGPUUtil << "%" << std::endl;
        }
        if(Config.isMeasureSMClk==true){
            avgSMClk= sumSMClk / TotalDuration;
            std::cout << "minSMClk: " << minSMClk << " MHz; avgSMClk: " << avgSMClk << " MHz; maxSMClk: " << maxSMClk << " MHz" << std::endl;
        }

        if(Config.isMeasureMemUtil==true || Config.isMeasureMemClk==true){
            std::cout << std::endl;
            std::cout << "-------- GPU Memory --------" << std::endl;
        }
        if(Config.isMeasureMemUtil==true){
            avgMemUtil = sumMemUtil / TotalDuration;
            std::cout << "minMemUtil: " << minMemUtil << " %; avgMemUtil: " << avgMemUtil << " %; maxMemUtil: " << maxMemUtil << " %" << std::endl;
        }
        if(Config.isMeasureMemClk==true){
            avgMemClk= sumMemClk / TotalDuration;
            std::cout << "minMemClk: " << minMemClk << " MHz; avgMemClk: " << avgMemClk << " MHz; maxMemClk: " << maxMemClk << " MHz" << std::endl;
        }

        
        if(Config.isGenOutFile==true){
            std::cout << "Write data to file..." << std::endl;
            // 这里以后再写
        }

        return 0;
    }
};



#endif