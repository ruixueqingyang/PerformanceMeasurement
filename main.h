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

pthread_cond_t condTStart; // control children processes to start simultaneously
pthread_mutex_t mutexTStart; // mutex for condTStart

int ChlidWaitCount;
int ChlidFinishCount; // Initial value is 0; ChlidFinishCount++ when a child thread/process finished; main thread check the value of ChlidFinishCount to know whether all child finished
pthread_cond_t condTEnd; // Child thread sends condTEnd to main thread when the child thread finished
pthread_mutex_t mutexTEnd; // mutex for condTStart

sem_t semPEnd; // semaphores control children processes of applications

#define VECTOR_RESERVE 20000

enum MEASURE_MODEL {
    INTERACTION,
    // DURATION,
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
    // float MeasureDuration; // (s) Sampling will keep for the specific measurement duration
    float PostInterval; // (s) Sampling will keep for the specific time interval after all applications are completed

    std::vector< char* > vecAppPath;
    std::vector< char** > vecAppAgrv;

    void init();
    int LoadAppList(char* AppListPath);

    CONFIG(){
        init();
    }
    ~CONFIG();
};

CONFIG::~CONFIG(){
    for(size_t i = 0; i < vecAppPath.size(); i++){
        if(vecAppPath[i] != NULL){
            free(vecAppPath[i]);
            vecAppPath[i] = NULL;
        }
    }

    for(size_t i = 0; i < vecAppAgrv.size(); i++){
        if(vecAppAgrv[i] != NULL){

            size_t j = 0;
            while(vecAppAgrv[i][j] != NULL){
                free(vecAppAgrv[i][j]);
                vecAppAgrv[i][j] = NULL;
                j++;
            }
            
            free(vecAppAgrv[i]);
            vecAppAgrv[i] = NULL;
        }
    }
}

int CONFIG::LoadAppList(char* AppListPath){

    // std::string src = AppListPath;

    std::ifstream srcStream(AppListPath, std::ifstream::in); // |std::ifstream::binary
    if(!srcStream.is_open()){
        srcStream.close();
        std::cerr << "ERROR: failed to open application list file" << std::endl;
        exit(1);
    }

    std::string TmpStr;
    const char* delim = " \r"; // 一行中 使用 空格 分词

    std::getline(srcStream, TmpStr);
    while(!TmpStr.empty()){ // 读取一行

        char* pCharBuff = (char*)malloc( sizeof(char) * (TmpStr.length()+1) );
        strcpy(pCharBuff, TmpStr.c_str());

        // 读取第一个 词, 即应用可执行文件路径
        char* TmpPtrChar = strtok(pCharBuff, delim);

        vecAppPath.emplace_back( (char*)malloc( sizeof(char) * (strlen(TmpPtrChar)+1) ) );
        strcpy(vecAppPath.back(), TmpPtrChar);

        std::vector<char*> TmpVecArg;

        // 这里要处理 TmpVecArg[0], 复制可执行文件/命令, 而不要前边的路径
        std::string TmpStr1 = vecAppPath.back();
        size_t found = TmpStr1.find_last_of('/');
        TmpStr1 = TmpStr1.substr(found+1);
        TmpVecArg.emplace_back( (char*)malloc( sizeof(char) * (TmpStr1.length()+1) ) );
        strcpy(TmpVecArg.back(), TmpStr1.c_str());

        while( ( TmpPtrChar = strtok(NULL, delim) ) ){
            TmpVecArg.emplace_back( (char*)malloc( sizeof(char) * (strlen(TmpPtrChar)+1) ) );
            strcpy(TmpVecArg.back(), TmpPtrChar);
        }

        vecAppAgrv.emplace_back( (char**)malloc( sizeof(char*)*(TmpVecArg.size()+1) ) );
        vecAppAgrv.back()[TmpVecArg.size()] = NULL;

        for(size_t i = 0; i < TmpVecArg.size(); i++){
            vecAppAgrv.back()[i] = TmpVecArg[i];
        }

        free(pCharBuff);
        TmpStr.clear();
        std::getline(srcStream, TmpStr);
    }
    
    srcStream.close();

    return 0;
}

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
    // MeasureDuration = -1.0;
    PostInterval = 0.0;
}

class PERF_DATA{
public:

    std::ofstream outStream;

    CUdevice cuDevice;
    nvmlDevice_t nvmlDevice;
    int ComputeCapablityMajor;
    int ComputeCapablityMinor;
    unsigned int long long SampleCount;
    double TotalDuration; // (s)

    struct timeval prevTimeStamp, currTimeStamp;
    std::vector<double> vecTimeStamp; // (s)
    double StartTimeStamp; // (s)
    
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
        StartTimeStamp = 0.0;
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

        vecTimeStamp.reserve(VECTOR_RESERVE);
        vecPower.reserve(VECTOR_RESERVE);
        vecMemUtil.reserve(VECTOR_RESERVE);
        vecMemClk.reserve(VECTOR_RESERVE);
        vecGPUUtil.reserve(VECTOR_RESERVE);
        vecSMClk.reserve(VECTOR_RESERVE);


        if(Config.isGenOutFile){
            outStream.open(Config.OutFilePath, std::ifstream::out);
            if(!outStream.is_open()){
                outStream.close();
                std::cerr << "ERROR: failed to open output file" << std::endl;
                return 1;
            }
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

        bool isOpen;

        if(Config.isGenOutFile==true){
            if(outStream.is_open()){
                isOpen = true;
            }else{
                outStream.close();
                std::cerr << "ERROR: output file is not opened" << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "-------- Performance Measurement Results --------" << std::endl;
        std::cout << "Actual Measurement Duration: " << TotalDuration << " s" << std::endl;
        std::cout << "Actual Sampling Count: " << SampleCount << std::endl;

        if(Config.isGenOutFile==true && isOpen==true){
            outStream << "-------- Performance Measurement Results --------" << std::endl;
            outStream << "Actual Measurement Duration: " << TotalDuration << " s" << std::endl;
            outStream << "Actual Sampling Count: " << SampleCount << std::endl;
        }
        

        if(Config.isMeasureEnergy==true){
            avgPower = Energy / TotalDuration;
            std::cout << std::endl;
            std::cout << "-------- Energy&Power (All) --------" << std::endl;
            std::cout << "Energy: " << Energy/1000 << " J" << std::endl;
            std::cout << "minPower: " << ((float)minPower)/1000 << " W; avgPower: " << avgPower/1000 << " W; maxPower: " << ((float)maxPower)/1000 << " W" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << std::endl;
                outStream << "-------- Energy&Power (All) --------" << std::endl;
                outStream << "Energy: " << Energy/1000 << " J" << std::endl;
                outStream << "minPower: " << ((float)minPower)/1000 << " W; avgPower: " << avgPower/1000 << " W; maxPower: " << ((float)maxPower)/1000 << " W" << std::endl;
            }
        }

        if(Config.isMeasureEnergy==true){
            avgPower = Energy / TotalDuration;
            float EnergyAbove = Energy/1000 - Config.PowerThreshold*TotalDuration;
            std::cout << std::endl;
            std::cout << "-------- Energy&Power (Above Threshold) --------" << std::endl;
            std::cout << "Power Threshold: " << Config.PowerThreshold << " W" << std::endl;
            std::cout << "Energy: " << EnergyAbove << " J" << std::endl;
            std::cout << "minPower: " << ((float)minPower)/1000-Config.PowerThreshold << " W; avgPower: " << avgPower/1000-Config.PowerThreshold << " W; maxPower: " << ((float)maxPower)/1000-Config.PowerThreshold << " W" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << std::endl;
                outStream << "-------- Energy&Power (Above Threshold) --------" << std::endl;
                std::cout << "Power Threshold: " << Config.PowerThreshold << " W" << std::endl;
                outStream << "Energy: " << EnergyAbove << " J" << std::endl;
                outStream << "minPower: " << ((float)minPower)/1000-Config.PowerThreshold << " W; avgPower: " << avgPower/1000-Config.PowerThreshold << " W; maxPower: " << ((float)maxPower)/1000-Config.PowerThreshold << " W" << std::endl;
            }
        }

        if(Config.isMeasureGPUUtil==true || Config.isMeasureSMClk==true){
            std::cout << std::endl;
            std::cout << "-------- GPU SM --------" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << std::endl;
                outStream << "-------- GPU SM --------" << std::endl;
            }
        }
        if(Config.isMeasureGPUUtil==true){
            avgGPUUtil = sumGPUUtil / TotalDuration;
            std::cout << "minGPUUtil: " << minGPUUtil << " %; avgGPUUtil: " << avgGPUUtil << " %; maxGPUUtil: " << maxGPUUtil << "%" << std::endl;
            
            if(Config.isGenOutFile==true && isOpen==true){
                outStream << "minGPUUtil: " << minGPUUtil << " %; avgGPUUtil: " << avgGPUUtil << " %; maxGPUUtil: " << maxGPUUtil << "%" << std::endl;
            }
        }
        if(Config.isMeasureSMClk==true){
            avgSMClk= sumSMClk / TotalDuration;
            std::cout << "minSMClk: " << minSMClk << " MHz; avgSMClk: " << avgSMClk << " MHz; maxSMClk: " << maxSMClk << " MHz" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << "minSMClk: " << minSMClk << " MHz; avgSMClk: " << avgSMClk << " MHz; maxSMClk: " << maxSMClk << " MHz" << std::endl;
            }
        }

        if(Config.isMeasureMemUtil==true || Config.isMeasureMemClk==true){
            std::cout << std::endl;
            std::cout << "-------- GPU Memory --------" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << std::endl;
                outStream << "-------- GPU Memory --------" << std::endl;
            }
        }
        if(Config.isMeasureMemUtil==true){
            avgMemUtil = sumMemUtil / TotalDuration;
            std::cout << "minMemUtil: " << minMemUtil << " %; avgMemUtil: " << avgMemUtil << " %; maxMemUtil: " << maxMemUtil << " %" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << "minMemUtil: " << minMemUtil << " %; avgMemUtil: " << avgMemUtil << " %; maxMemUtil: " << maxMemUtil << " %" << std::endl;
            }
        }
        if(Config.isMeasureMemClk==true){
            avgMemClk= sumMemClk / TotalDuration;
            std::cout << "minMemClk: " << minMemClk << " MHz; avgMemClk: " << avgMemClk << " MHz; maxMemClk: " << maxMemClk << " MHz" << std::endl;

            if(Config.isGenOutFile==true && isOpen==true){
                outStream << "minMemClk: " << minMemClk << " MHz; avgMemClk: " << avgMemClk << " MHz; maxMemClk: " << maxMemClk << " MHz" << std::endl;
            }
        }

        
        if(Config.isGenOutFile==true && isOpen==true){
            std::cout << "Write raw data to file..." << std::endl;
        }else{
            return 0;
        }

        outStream << std::endl;
        outStream << "-------- Raw Data --------" << std::endl;
        outStream << std::endl;

        // outStream << "vecTimeStamp size: " << vecTimeStamp.size() << std::endl;
        // outStream << "vecPower size: " << vecPower.size() << std::endl;
        // outStream << "vecGPUUtil size: " << vecGPUUtil.size() << std::endl;
        // outStream << "vecSMClk size: " << vecSMClk.size() << std::endl;
        // outStream << "vecMemUtil size: " << vecMemUtil.size() << std::endl;
        // outStream << "vecMemClk size: " << vecMemClk.size() << std::endl;
        // output Power Threshold
        if(Config.isMeasureEnergy==true){
            outStream << "Power Threshold: " << Config.PowerThreshold << " W" << std::endl;
        }
        outStream << std::endl;

        // output data which were sample and their order
        outStream << "Time Stamp (s)" << std::endl;
        if(Config.isMeasureEnergy==true){
            outStream << "Power (W)" << std::endl;
        }
        if(Config.isMeasureGPUUtil==true){
            outStream << "GPUUtil (%)" << std::endl;
        }
        if(Config.isMeasureSMClk==true){
            outStream << "SMClk (MHz)" << std::endl;
        }
        if(Config.isMeasureMemUtil==true){
            outStream << "MemUtil (%)" << std::endl;
        }
        if(Config.isMeasureMemClk==true){
            outStream << "MemClk (MHz)" << std::endl;
        }
        
        // output ram data
        outStream << std::endl;
        for(size_t i = 0; i < vecTimeStamp.size(); i++){
            
            outStream << vecTimeStamp[i] << std::endl;
            if(Config.isMeasureEnergy==true){
                outStream << float(vecPower[i])/1000 << std::endl;
            }
            if(Config.isMeasureGPUUtil==true){
                outStream << vecGPUUtil[i] << std::endl;
            }
            if(Config.isMeasureSMClk==true){
                outStream << vecSMClk[i] << std::endl;
            }
            if(Config.isMeasureMemUtil==true){
                outStream << vecMemUtil[i] << std::endl;
            }
            if(Config.isMeasureMemClk==true){
                outStream << vecMemClk[i] << std::endl;
            }
            outStream << std::endl;

        }

        return 0;
    }
};



#endif