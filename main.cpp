/*******************************************************************************
Copyright(C), 2020-2020, 瑞雪轻飏
     FileName: main.cpp
       Author: 瑞雪轻飏
      Version: 0.01
Creation Date: 20200504
  Description: 性能/能耗测量工具的主/入口 cpp 文件, 包含 main 函数
       Others: 
*******************************************************************************/

#include "main.h"

// PerformanceMeasurement.bin -h
// PerformanceMeasurement.bin

CONFIG Config;
PERF_DATA PerfData;
POWER_MANAGER PM;

int ParseOptions(int argc, char** argv);
int MeasureInit();
void AlarmSampler(int signum);
static void* ForkChildProcess(void* arg);

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

    PM.Set();

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
    }else if(Config.MeasureModel==MEASURE_MODEL::APPLICATION){

        std::vector<pthread_t> vecAppTid;
        std::vector<unsigned int> vecAppIndex;
        vecAppTid.reserve(Config.vecAppPath.size());
        vecAppIndex.reserve(Config.vecAppPath.size());
        // int count = 0;
        pthread_attr_t attr;

        // 初始化条件变量 初值为 0
        pthread_mutex_init(&mutexTStart, NULL);
        pthread_cond_init(&condTStart, NULL);
        // pthread_mutex_lock(&mutexTStart);

        ChlidFinishCount = 0;
        ChlidWaitCount = 0;
        pthread_mutex_init(&mutexTEnd, NULL);
        pthread_cond_init(&condTEnd, NULL);

        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        // fork 所需的所有子线程, 并检查是否出错, 并记录子线程 tid
        for(unsigned int i = 0; i<Config.vecAppPath.size(); i++){
            vecAppIndex[i] = i;
            err = pthread_create(&vecAppTid[i], &attr, ForkChildProcess, (void*)&vecAppIndex[i]);
            if(err != 0) {
                std::cerr << "ERROR: pthread_create() return code: " << err << std::endl;
                return -1;
            }
        }

        while(ChlidWaitCount != Config.vecAppPath.size()){
            usleep(10000);
        }

        // 启动采样
        signal(SIGALRM, AlarmSampler);
        ualarm(10, Config.SampleInterval*1000);

        std::cout << "Sampling has already started." << std::endl;
        std::cout << "Sampling..." << std::endl;

        // 启动子线程, 创建应用进程
        pthread_cond_broadcast(&condTStart);
        pthread_mutex_unlock(&mutexTStart);

        while(true){
            pthread_mutex_lock(&mutexTEnd);

            pthread_cond_wait(&condTEnd, &mutexTEnd); // 这里会先解锁 mutexTEnd, 然后阻塞, 返回后再上锁 mutexTEnd

            // 如果所有子线程都完成了, 即所有应用都执行完了
            if(ChlidFinishCount == Config.vecAppPath.size()){
                pthread_mutex_unlock(&mutexTEnd); // 为了上锁/解锁配对, 加上这句
                break;
            }

            pthread_mutex_unlock(&mutexTEnd);
        }

        // 结束采样
        ualarm(0, Config.SampleInterval*1000);
        std::cout << "Sampling has already stopped." << std::endl;
        
        for (unsigned int i = 0; i<Config.vecAppPath.size(); i++) {
            pthread_join(vecAppTid[i], NULL);
        } 
        
        // 清理 并 退出
        pthread_attr_destroy(&attr);
        pthread_mutex_destroy(&mutexTStart);
        pthread_mutex_destroy(&mutexTEnd);
        pthread_cond_destroy(&condTStart);
        pthread_cond_destroy(&condTEnd);
        // pthread_exit(NULL);

        // output result
        PerfData.output(Config);
        
    }else{
        std::cerr << "Illegal measurement mode !" << std::endl;
    }
    
    PM.Reset();
    
    return 0;
}

static void* ForkChildProcess(void* arg){

    pid_t AppPid;
    int PStatus;

    unsigned int* pAppIndex = (unsigned int*)arg;

    pthread_mutex_lock(&mutexTStart);
    ChlidWaitCount++;
    pthread_cond_wait(&condTStart, &mutexTStart);
    pthread_mutex_unlock(&mutexTStart);

    AppPid = fork();
    if(AppPid == 0){ // 子进程

        std::cout << "Application " << *pAppIndex + 1 << " is being launched..." << std::endl;
        execv (Config.vecAppPath[*pAppIndex], Config.vecAppAgrv[*pAppIndex]);
        printf("ERROR: execv on application %s failed\n", Config.vecAppPath[0]);
        exit(-1);

    }else if(AppPid < 0){ // 主进程 fork 出错
        std::cerr << "ERROR: fork on application " << *pAppIndex + 1 << " failed" << std::endl;
        exit(-1);
    }

    // 主进程
    waitpid(AppPid,&PStatus,0); // 等待应用所在进程运行结束
    
    pthread_mutex_lock(&mutexTEnd);
    std::cout << "Application " << *pAppIndex + 1 << " has finished" << std::endl;
    ChlidFinishCount++; // 应用完成计数 ++
    pthread_cond_signal(&condTEnd); // 通知主线程, 一个应用已经执行完成
    pthread_mutex_unlock(&mutexTEnd);

    pthread_exit(NULL);
}

void AlarmSampler(int signum){

    if(signum != SIGALRM) return;

    nvmlReturn_t nvmlResult;

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

        PerfData.StartTimeStamp = (double)PerfData.currTimeStamp.tv_sec + (double)PerfData.currTimeStamp.tv_usec * 1e-6;

        if(Config.isGenOutFile==true){
            PerfData.vecTimeStamp.clear();
            PerfData.vecTimeStamp.push_back(0.0);
        }

    }else{
        double ActualInterval; // (s)
        double RelativeTimeStamp; // (s)

        ActualInterval = (double)(PerfData.currTimeStamp.tv_sec - PerfData.prevTimeStamp.tv_sec) + (double)(PerfData.currTimeStamp.tv_usec - PerfData.prevTimeStamp.tv_usec) * 1e-6;

        RelativeTimeStamp = (double)PerfData.currTimeStamp.tv_sec + (double)PerfData.currTimeStamp.tv_usec * 1e-6 - PerfData.StartTimeStamp;

        PerfData.TotalDuration = RelativeTimeStamp;

        if(Config.isGenOutFile==true){
            PerfData.vecTimeStamp.push_back(RelativeTimeStamp);
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

int ParseOptions(int argc, char** argv){
    
    int err = 0;
    unsigned int indexArg = 1; // 当前 flag(即-xxxx) 在 argv 中的 index
    extern int optind,opterr,optopt;
    extern char *optarg;
    const char usage[] = "Usage: %s [-e] \nType '??? -h' for help.\n";

    Config.init();
    PM.init();

    //定义长选项
    static struct option long_options[] = 
    {
        {"h", no_argument, NULL, 'h'},
        {"help", no_argument, NULL, 'h'},

        // {"m", required_argument, NULL, 'm'},
        // {"model", required_argument, NULL, 'm'},

        // {"d", required_argument, NULL, 'd'},
        // {"duration", required_argument, NULL, 'd'},
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

        {"e", no_argument, NULL, 'e'},
        {"energy", no_argument, NULL, 'e'},
        {"memuti", no_argument, NULL, 1},
        {"memclk", no_argument, NULL, 2},
        {"gpuuti", no_argument, NULL, 3},
        {"smclk", no_argument, NULL, 4},

        {"tune", required_argument, NULL, 0},
        {"tunearg", required_argument, NULL, 5}
    };

    int aSet=0, lSet=0, pSet=0, oSet=0, iSet=0, sSet=0, tSet=0, eSet=0, memutiSet=0, memclkSet=0, gpuutiSet=0, smclkSet=0, tuneSet=0, tuneargSet=0;

    int c = 0; //用于接收选项
    /*循环处理参数*/
    while(EOF != (c = getopt_long_only(argc, argv, "", long_options, NULL))){
        if(aSet!=0){ // -a 参数必须放在最后, 之后的参数都看作是应用的参数
            break;
        }
        //打印处理的参数
        //printf("start to process %d para\n",optind);
        switch(c){
            case 'h':
                printf ("这里应该打印帮助信息...\n");
                //printf ( HELP_INFO );
                indexArg++;
                break;
            case 'a':
            {
                if(aSet!=0 || lSet!=0){
                    std::cerr << "WARNING: -a/-app/-l/-applistfile is set multiple times, the first value is used" << std::endl;
                    break;
                }
                aSet++;
                if(argc-indexArg-1 <= 0){ // 缺少应用参数
                    std::cerr << "ERROR: flag -a/-app need application path" << std::endl;
                    err |= 1;
                    return err;
                }

                Config.MeasureModel=MEASURE_MODEL::APPLICATION;

                Config.vecAppPath.emplace(Config.vecAppPath.begin(), (char*)NULL);
                Config.vecAppPath[0] = (char*)malloc( sizeof(char) * (strlen(optarg)+1) );
                strcpy(Config.vecAppPath[0], optarg);
                
                Config.vecAppAgrv.emplace(Config.vecAppAgrv.begin(), (char**)NULL);
                Config.vecAppAgrv[0] = (char**)malloc( sizeof(char*) * (argc-indexArg) );
                Config.vecAppAgrv[0][argc-indexArg-1] = NULL;

                // 这里要处理 Config.vecAppAgrv[0][0], 复制可执行文件/命令, 而不要前边的路径
                std::string TmpString = optarg;
                size_t found = TmpString.find_last_of('/');
                TmpString = TmpString.substr(found+1);
                Config.vecAppAgrv[0][0] = (char*)malloc( sizeof(char) * (TmpString.length()+1) );
                strcpy(Config.vecAppAgrv[0][0], TmpString.c_str());
                
                for (size_t i = indexArg+2; i < argc; i++) {
                    Config.vecAppAgrv[0][i-indexArg-1] = (char*)malloc( sizeof(char) * (strlen(argv[i])+1) );
                    strcpy(Config.vecAppAgrv[0][i-indexArg-1], argv[i]);
                }
            }
                break;
            
            case 'l':
                if(aSet!=0 || lSet!=0){
                    std::cerr << "WARNING: -a/-app/-l/-applistfile is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                lSet++;
                if(argc-indexArg-1 <= 0){ // 缺少应用参数
                    std::cerr << "ERROR: flag \"-a\" need application path" << std::endl;
                    err |= 1;
                    return err;
                }
                Config.MeasureModel=MEASURE_MODEL::APPLICATION;
                err = Config.LoadAppList(optarg);
                indexArg+=2;
                break;
            case 'p':
                if(pSet!=0){
                    std::cerr << "WARNING: -p/-postinterval is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                pSet++;
                Config.PostInterval = atof(optarg);
                indexArg+=2;
                break;
            case 'o':
                if(oSet!=0){
                    std::cerr << "WARNING: -o/-outfile is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                oSet++;
                Config.isGenOutFile = true;
                Config.OutFilePath = optarg;
                indexArg+=2;
                break;
            case 'i':
                if(iSet!=0){
                    std::cerr << "WARNING: -i/-id is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                iSet++;
                Config.DeviceID = atoi(optarg);
                PM.indexGPU = Config.DeviceID;

                if(Config.DeviceID < 0){
                    std::cerr << "ERROR: -i/-id value illegal: " << optarg << std::endl;
                    err |= 1;
                }

                indexArg+=2;
                break;
            case 's':
                if(sSet!=0){
                    std::cerr << "WARNING: -s/-samplinginterval is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                sSet++;
                Config.SampleInterval = atof(optarg);
                indexArg+=2;
                break;
            case 't':
                if(tSet!=0){
                    std::cerr << "WARNING: -t/-threshold is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                tSet++;
                Config.PowerThreshold = atof(optarg);
                indexArg+=2;
                break;
            case 'e':
                if(eSet!=0){
                    std::cerr << "WARNING: -e/-energy is set multiple times" << std::endl;
                    indexArg++;
                    break;
                }
                eSet++;
                Config.isMeasureEnergy = true;
                indexArg++;
                break;
            case 1:
                if(memutiSet!=0){
                    std::cerr << "WARNING: -memuti is set multiple times" << std::endl;
                    indexArg++;
                    break;
                }
                memutiSet++;
                Config.isMeasureMemUtil = true;
                indexArg++;
                break;
            case 2:
                if(memclkSet!=0){
                    std::cerr << "WARNING: -memclk is set multiple times" << std::endl;
                    indexArg++;
                    break;
                }
                memclkSet++;
                Config.isMeasureMemClk = true;
                indexArg++;
                break;
            case 3:
                if(gpuutiSet!=0){
                    std::cerr << "WARNING: -gpuuti is set multiple times" << std::endl;
                    indexArg++;
                    break;
                }
                gpuutiSet++;
                Config.isMeasureGPUUtil = true;
                indexArg++;
                break;
            case 4:
                if(smclkSet!=0){
                    std::cerr << "WARNING: -smclk is set multiple times" << std::endl;
                    indexArg++;
                    break;
                }
                smclkSet++;
                Config.isMeasureSMClk = true;
                indexArg++;
                break;

            case 0:
                if(tuneSet!=0){
                    std::cerr << "WARNING: -tune is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                tuneSet++;
                if(strcmp("DVFS", optarg)==0){
                    PM.TuneType = 0;
                }else if(strcmp("POWER", optarg)==0){
                    PM.TuneType = 1;
                }else{
                    std::cerr << "Power Manager ERROR: Invalid tuning technology type (-tune " << optarg << ")." << std::endl;
                    err |= 1;
                }
                indexArg+=2;
                break;
            case 5:
                if(tuneargSet!=0){
                    std::cerr << "WARNING: -tunearg is set multiple times, the first value is used" << std::endl;
                    indexArg+=2;
                    break;
                }
                tuneargSet++;
                PM.TuneArg = atoi(optarg);
                if(PM.TuneArg < 0){
                    std::cerr << "ERROR: -tunearg value illegal: " << optarg << std::endl;
                    err |= 1;
                }
                indexArg+=2;
                break;
            
            //表示选项不支持
            case '?':
                printf("unknow option:%c\n", optopt);
                err |= 1;
                indexArg++;
                break;
            default:
                break;
        }  
    }

    err |= PM.initArg();

    return err;
}