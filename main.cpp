/*******************************************************************************
Copyright(C), 1992-2020, 瑞雪轻飏
     FileName: main.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20200504
  Description: 性能/能耗测量工具的主/入口 cpp 文件, 包含 main 函数
       Others: 
*******************************************************************************/

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
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>

// #include <cuda/cuda.h>
// #include <cuda/cuda_runtime_api.h>

#define VECTOR_RESERVE 20000


// PerformanceMeasurement.bin -h
// PerformanceMeasurement.bin
// PerformanceMeasurement.bin -c config_dir

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

// double times[SAMPLE_MAX_SIZE_DEFAULT];
// // lld elapsedTimes[SAMPLE_MAX_SIZE_DEFAULT];
// unsigned int powers[SAMPLE_MAX_SIZE_DEFAULT];
// nvmlPstates_t pStates[SAMPLE_MAX_SIZE_DEFAULT];
// unsigned int gpuUtil[SAMPLE_MAX_SIZE_DEFAULT];
// unsigned int memoryUtil[SAMPLE_MAX_SIZE_DEFAULT];

CONFIG Config;
PERF_DATA PerfData;

int ParseOptions(int argc, char** argv);
int MeasureInit();
void AlarmSampler(int signum);

int ParseOptions(int argc, char** argv){
    
    int err = 0;
    int index;
    extern int optind,opterr,optopt;
    extern char *optarg;
    const char usage[] = "Usage: %s [-e] \nType '??? -h' for help.\n";

    Config.init();

    //定义长选项
    static struct option long_options[] = 
    {
        {"h", no_argument, NULL, 'h'},
        {"help", no_argument, NULL, 'h'},

        {"m", required_argument, NULL, 'm'},
        {"model", required_argument, NULL, 'm'},

        {"d", required_argument, NULL, 'd'},
        {"duration", required_argument, NULL, 'd'},
        {"a", required_argument, NULL, 'a'},
        {"app", required_argument, NULL, 'a'},
        {"l", required_argument, NULL, 'l'},
        {"applistfile", required_argument, NULL, 'l'},
        {"p", required_argument, NULL, 'p'},
        {"postinterval", required_argument, NULL, 'p'},
        
        {"o", required_argument, NULL, 'o'},
        {"outfile", required_argument, NULL, 'o'},

        {"i", required_argument, NULL, 'i'},
        {"id", required_argument, NULL, 'i'},
        {"s", required_argument, NULL, 's'},
        {"samplinginterval", required_argument, NULL, 's'},
        {"t", required_argument, NULL, 't'},
        {"threshold", required_argument, NULL, 't'},

        {"e", required_argument, NULL, 'e'},
        {"energy", required_argument, NULL, 'e'},
        {"memuti", required_argument, NULL, 1},
        {"memclk", required_argument, NULL, 2},
        {"gpuuti", required_argument, NULL, 3},
        {"smclk", required_argument, NULL, 4}
    };

    int c = 0; //用于接收选项
    /*循环处理参数*/
    while(EOF != (c = getopt_long_only(argc, argv, "", long_options, &index)))
    {
        //打印处理的参数
        //printf("start to process %d para\n",optind);
        switch(c)
        {
            case 'h':
                printf ("这里应该打印帮助信息...\n");
                //printf ( HELP_INFO );
                break;
            case 'm':
                if(0 == strcmp("ITR", optarg)){
                    Config.MeasureModel=MEASURE_MODEL::INTERACTION;
                }else if(0 == strcmp("DUR", optarg)){
                    Config.MeasureModel=MEASURE_MODEL::DURATION;
                    std::cerr << "DUR 模式还没实现!" << std::endl;
                    err |= 1;
                }if(0 == strcmp("APP", optarg)){
                    Config.MeasureModel=MEASURE_MODEL::APPLICATION;
                    std::cerr << "APP 模式还没实现!" << std::endl;
                    err |= 1;
                }else{
                    fprintf ( stderr, "%s: error: -m/-model value illegal.\n", optarg );
                    err |= 1;
                }                
                break;
            case 'd':
                Config.MeasureDuration = atof(optarg);
                break;
            case 'a':
                // if(0 == strlen())
                // std::string ConfigDir = optarg;
                // err |= Config.init(ConfigDir);
                std::cerr << "该功能还没实现!" << std::endl;
                err |= 1;
                break;
            case 'l':
                // std::string ConfigDir = optarg;
                // err |= Config.init(ConfigDir);
                std::cerr << "该功能还没实现!" << std::endl;
                err |= 1;
                break;
            case 'p':
                Config.PostInterval = atof(optarg);
                break;
            case 'o':
                Config.isGenOutFile = true;
                Config.OutFilePath = optarg;
                break;
            case 'i':
                Config.DeviceID = atoi(optarg);
                break;
            case 's':
                Config.SampleInterval = atof(optarg);
                break;
            case 't':
                Config.PowerThreshold = atof(optarg);
                break;
            case 'e':
                if(0 == atoi(optarg)){
                    Config.isMeasureEnergy = false;
                }else{
                    Config.isMeasureEnergy = true;
                }
                break;
            case 1:
                if(0 == atoi(optarg)){
                    Config.isMeasureMemUtil = false;
                }else{
                    Config.isMeasureMemUtil = true;
                }
                break;
            case 2:
                if(0 == atoi(optarg)){
                    Config.isMeasureMemClk = false;
                }else{
                    Config.isMeasureMemClk = true;
                }
                break;
            case 3:
                if(0 == atoi(optarg)){
                    Config.isMeasureGPUUtil = false;
                }else{
                    Config.isMeasureGPUUtil = true;
                }
                break;
            case 4:
                if(0 == atoi(optarg)){
                    Config.isMeasureSMClk = false;
                }else{
                    Config.isMeasureSMClk = true;
                }
                break;
            
            //表示选项不支持
            case '?':
                printf("unknow option:%c\n",optopt);
                err |= 1;
                break;
            default:
                break;
        }  
    }

    return 0;
}

// 主函数
int main(int argc, char** argv){
    int err;
    char CharBuffer[6];
    std::string tmpString;
    const std::string stop = "stop";
    const std::string Stop = "Stop";
    const std::string STOP = "STOP";

    // 处理输入参数
    // Parse input arguments
    err = ParseOptions(argc, argv);
    if( 0 != err ) exit(err);

    err = MeasureInit();
    if( 0 != err ) exit(err);

    if(Config.MeasureModel==MEASURE_MODEL::INTERACTION){
        signal(SIGALRM, AlarmSampler);
        ualarm(10, Config.SampleInterval*1000);

        std::cout << "Sampling has already started." << std::endl;
        std::cout << "Sampling..." << std::endl;
        std::cout << "Type \"stop\" to stop sampling: ";

        while(true){
            std::cin.getline(CharBuffer, 6);
            CharBuffer[5] = '\0';
            tmpString = CharBuffer;
            // std::getline(std::cin, tmpString);
            if( tmpString==stop || tmpString==Stop || tmpString==STOP){
                ualarm(0, Config.SampleInterval*1000);
                std::cout << "Sampling has already stopped." << std::endl;
                break;
            }else{
                std::cout << "Sampling..." << std::endl;
                std::cout << "Type \"stop\" to stop sampling: ";
            }
        }

        // output result
        PerfData.output(Config);
    }else if(Config.MeasureModel==MEASURE_MODEL::DURATION){
        std::cerr << "DURATION 模式还没实现!" << std::endl;
    }else if(Config.MeasureModel==MEASURE_MODEL::APPLICATION){
        std::cerr << "APPLICATION 模式还没实现!" << std::endl;
    }else{
        std::cerr << "Illegal measurement mode !" << std::endl;
    }
    
    
    return 0;
}

void AlarmSampler(int signum){

    if(signum != SIGALRM) return;

    nvmlReturn_t nvmlResult;
    // int i;
    // struct timeval time;
    // double now;
    // static double before = 0;

    double ActualInterval; // (s)

    gettimeofday(&PerfData.currTimeStamp,NULL);

    // get current value
    if(Config.isMeasureEnergy==true){
        nvmlResult = nvmlDeviceGetPowerUsage(PerfData.nvmlDevice, &PerfData.currPower);
    }
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to get power usage: %s\n", nvmlErrorString(nvmlResult));
        exit(-1);
    }

    if(Config.isMeasureSMClk==true){
        nvmlResult = nvmlDeviceGetClockInfo(PerfData.nvmlDevice, NVML_CLOCK_SM, &PerfData.currSMClk);
    }
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to get SM clock: %s\n", nvmlErrorString(nvmlResult));
        exit(-1);
    }

    if(Config.isMeasureMemClk==true){
        nvmlResult = nvmlDeviceGetClockInfo(PerfData.nvmlDevice, NVML_CLOCK_MEM, &PerfData.currMemClk);
    }
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to get memory clock: %s\n", nvmlErrorString(nvmlResult));
        exit(-1);
    }

    if(Config.isMeasureMemUtil==true || Config.isMeasureGPUUtil==true){
        nvmlResult = nvmlDeviceGetUtilizationRates(PerfData.nvmlDevice, &PerfData.currUtil);
    }
    if (NVML_SUCCESS != nvmlResult) {
        printf("Failed to get utilization rate: %s\n", nvmlErrorString(nvmlResult));
        exit(-1);
    }

    // update SampleCount
    PerfData.SampleCount++;

    // update min/max and push data
    if(Config.isMeasureEnergy==true){
        if(PerfData.currPower < PerfData.minPower){
            PerfData.minPower = PerfData.currPower;
        }
        if(PerfData.currPower > PerfData.maxPower){
            PerfData.maxPower = PerfData.currPower;
        }
        if(Config.isGenOutFile==true){
            PerfData.vecPower.push_back(PerfData.currPower);
        }
    }

    if(Config.isMeasureSMClk==true){
        if(PerfData.currSMClk < PerfData.minSMClk){
            PerfData.minSMClk = PerfData.currSMClk;
        }
        if(PerfData.currSMClk > PerfData.maxSMClk){
            PerfData.maxSMClk = PerfData.currSMClk;
        }
        if(Config.isGenOutFile==true){
            PerfData.vecSMClk.push_back(PerfData.currSMClk);
        }
    }

    if(Config.isMeasureMemClk==true){
        if(PerfData.currMemClk < PerfData.minMemClk){
            PerfData.minMemClk = PerfData.currMemClk;
        }
        if(PerfData.currMemClk > PerfData.maxMemClk){
            PerfData.maxMemClk = PerfData.currMemClk;
        }
        if(Config.isGenOutFile==true){
            PerfData.vecMemClk.push_back(PerfData.currMemClk);
        }
    }

    if(Config.isMeasureMemUtil==true){
        if(PerfData.currUtil.memory < PerfData.minMemUtil){
            PerfData.minMemUtil = PerfData.currUtil.memory;
        }
        if(PerfData.currUtil.memory > PerfData.maxMemUtil){
            PerfData.maxMemUtil = PerfData.currUtil.memory;
        }
        if(Config.isGenOutFile==true){
            PerfData.vecMemUtil.push_back(PerfData.currUtil.memory);
        }
    }

    if(Config.isMeasureGPUUtil==true){
        if(PerfData.currUtil.gpu < PerfData.minGPUUtil){
            PerfData.minGPUUtil = PerfData.currUtil.gpu;
        }
        if(PerfData.currUtil.gpu > PerfData.maxGPUUtil){
            PerfData.maxGPUUtil = PerfData.currUtil.gpu;
        }
        if(Config.isGenOutFile==true){
            PerfData.vecGPUUtil.push_back(PerfData.currUtil.gpu);
        }
    }

    // calculate ActualInterval/sum value, and push timestamp
    if(PerfData.isFisrtSample==true){
        PerfData.isFisrtSample = false;

        if(Config.isGenOutFile==true){
            PerfData.vecTimeStamp.push_back(0.0);
        }

    }else{

        ActualInterval = (double)(PerfData.currTimeStamp.tv_sec - PerfData.prevTimeStamp.tv_sec) + (double)(PerfData.currTimeStamp.tv_usec - PerfData.prevTimeStamp.tv_usec) * 1e-6;

        PerfData.TotalDuration += ActualInterval;

        if(Config.isGenOutFile==true){
            PerfData.vecTimeStamp.push_back(ActualInterval);
        }
        if(Config.isMeasureEnergy==true){
            PerfData.Energy += (float)(PerfData.prevPower + PerfData.currPower) / 2 * ActualInterval;
        }
        if(Config.isMeasureSMClk==true){
            PerfData.sumSMClk += (float)(PerfData.prevSMClk + PerfData.currSMClk) / 2 * ActualInterval;
        }
        if(Config.isMeasureMemClk==true){
            PerfData.sumMemClk += (float)(PerfData.prevMemClk + PerfData.currMemClk) / 2 * ActualInterval;
        }
        if(Config.isMeasureMemUtil==true){
            PerfData.sumMemUtil += (float)(PerfData.prevUtil.memory + PerfData.currUtil.memory) / 2 * ActualInterval;
        }
        if(Config.isMeasureGPUUtil==true){
            PerfData.sumGPUUtil += (float)(PerfData.prevUtil.gpu + PerfData.currUtil.gpu) / 2 * ActualInterval;
        }
    }

    // update previous value
    PerfData.prevTimeStamp.tv_sec = PerfData.currTimeStamp.tv_sec;
    PerfData.prevTimeStamp.tv_usec = PerfData.currTimeStamp.tv_usec;
    PerfData.prevUtil.memory = PerfData.currUtil.memory;
    PerfData.prevUtil.gpu = PerfData.currUtil.gpu;
    PerfData.prevPower = PerfData.currPower;
    PerfData.prevMemClk = PerfData.currMemClk;
    PerfData.prevSMClk = PerfData.currSMClk;

}

int MeasureInit(){
    CUresult cuResult;
    cudaError_t cuError;
    nvmlReturn_t nvmlResult;
	int DeviceCount;
    //int a;

    PerfData.init(Config);

    cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        printf("Error code %d on cuInit\n", cuResult);
        exit(-1);
    }

    cuError = cudaGetDeviceCount(&DeviceCount);
    if (cuError != cudaSuccess) {
        printf("Error code %d on cudaGetDeviceCount\n", cuResult);
        exit(-1);
    }

    printf("Found %d device%s\n\n", DeviceCount, DeviceCount != 1 ? "s" : "");
    if (Config.DeviceID >= DeviceCount) {
        printf("DeviceID is out of range.\n");
        return -1;
    }

    cuResult = cuDeviceGet(&PerfData.cuDevice, Config.DeviceID);
    if (cuResult != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGet\n", cuResult);
        exit(-1);
    }

    cuResult = cuDeviceGetAttribute (&PerfData.ComputeCapablityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, PerfData.cuDevice);
    if (cuResult != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGetAttribute\n", cuResult);
        exit(-1);
    }
    cuResult = cuDeviceGetAttribute (&PerfData.ComputeCapablityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, PerfData.cuDevice);
    if (cuResult != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGetAttribute\n", cuResult);
        exit(-1);
    }

	// i_begin = -1;
	// i_end = -1;

	// terminate = 0;
	// begin_interval = 0;
	// end_interval = 0;

	// NVML INITIALIZATIONS
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvmlResult));

        printf("Press ENTER to continue...\n");
        getchar();
        return -1;
    }

	// nvmlResult = nvmlDeviceGetCount(&DeviceCount);
    // if (NVML_SUCCESS != nvmlResult)
    // {
    //     printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
    //     return -1;
    // }

	// printf("Found %d device%s\n\n", DeviceCount, DeviceCount != 1 ? "s" : "");
    // if (Config.DeviceID >= DeviceCount) {
    //     printf("Device_id is out of range.\n");
    //     return -1;
    // }
	nvmlResult = nvmlDeviceGetHandleByIndex(Config.DeviceID, &PerfData.nvmlDevice);
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to get handle for device 1: %s\n", nvmlErrorString(nvmlResult));
		 return -1;
	}
	// nvmlDeviceGetApplicationsClock  ( nvmlDevice, NVML_CLOCK_GRAPHICS, &core_clock);

    // nvmlDeviceGetApplicationsClock  ( nvmlDevice, NVML_CLOCK_SM, &clock);
    // printf("Current SM clock: %d\n", clock);
    // nvmlDeviceGetApplicationsClock  ( nvmlDevice, NVML_CLOCK_MEM, &mem_clock);


	//LAUNCH POWER SAMPLER
	// a = pthread_create(&thread_sampler, NULL, threadWork, NULL);
	// if(a) {
	// 	fprintf(stderr,"Error - pthread_create() return code: %d\n",a);
	// 	return -1;
	// }

	return 0;
}