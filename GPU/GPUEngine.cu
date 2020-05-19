/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../Timer.h"

//#include "GPUSptable.h"
#include "GPUMath.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint64_t *Tkangaroos, uint64_t *Wkangaroos, uint64_t DPmodule, uint32_t hop_modulo, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x*GPU_GRP_SIZE) * 12; // px[4] , py[4] , tk[4]
    
  ComputeKeys(Tkangaroos + xPtr, Wkangaroos + xPtr, DPmodule, hop_modulo, maxFound, found);

}

// ---------------------------------------------------------------------------------------

using namespace std;

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1} };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey, uint32_t pow2w, uint64_t totalK, uint32_t hop_modulo, int power, int fixedDP) { 

  // Initialise CUDA
  this->nbThreadPerGroup = nbThreadPerGroup;
  this->rekey = rekey;
  initialised = false;
  cudaError_t err;
  
  int pow2dp;  
  
  // Set fixed DP
  //fixedDP = 20;
  
  if (hop_modulo > NB_JUMP) {
	hop_modulo = (uint32_t)NB_JUMP;
  }	
  
  if (fixedDP > 0) {
	pow2dp = fixedDP;
	printf("GPUEngine: Fixed DPmodule = 2^%d \n", pow2dp);
  } else {
	//pow2dp = ((pow2w-(2*power))/2)-2;
	pow2dp = (int)((double)pow2w / 2.0 - log2((double)totalK) - 2);
	printf("GPUEngine: Optimal DPmodule = 2^%d \n", pow2dp);
  }
  
  printf("GPUEngine: Total kangaroos and their close relatives %lu ;-)\n", totalK);
  
  if (pow2dp > 24) {
	printf("GPUEngine: Old DPmodule = 2^%d \n", pow2dp);
	pow2dp = 24;
	printf("GPUEngine: New DPmodule = 2^%d \n", pow2dp);
  }
  if (pow2dp < 14) {
	printf("GPUEngine: Old DPmodule = 2^%d \n", pow2dp);
	pow2dp = 14;
	printf("GPUEngine: New DPmodule = 2^%d \n", pow2dp);
  }
  
  uint64_t DPmodule = (uint64_t)1 << pow2dp;
  
  if (fixedDP > 0) {
	printf("GPUEngine: Fixed DPmodule: 0x%lx 2^%d Hop_modulo: %d Power: %d \n", DPmodule, pow2dp, hop_modulo, power);
  } else {
	//printf("GPUEngine: DPmodule: 0x%lx 2^%d ((pow2W-(2*Power))/2)-2 Hop_modulo: %d Power: %d \n", DPmodule, pow2dp, hop_modulo, power);
	printf("GPUEngine: DPmodule: 0x%lx 2^%d (pow2W/2)-log(Kangaroos)-2 Hop_modulo: %d Power: %d \n", DPmodule, pow2dp, hop_modulo, power);
  }
  
  this->DPmodule = DPmodule;
  this->hop_modulo = hop_modulo;
  
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);  

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);
  
  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
  nbThread / nbThreadPerGroup,
  nbThreadPerGroup);
  deviceName = std::string(tmp);
  
  
  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("\nGPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }
  
  /*
  size_t stackSize = 49152;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("\nGPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }
  */
  
  // Allocate memory
  inputKangaroo = NULL;
  inputKangarooPinned = NULL;
  w_inputKangaroo = NULL;
  w_inputKangarooPinned = NULL;
  outputPrefix = NULL;
  outputPrefixPinned = NULL;
  jumpPinned = NULL;

  // Input kangaroos
  kangarooSize = nbThread * GPU_GRP_SIZE * 96;
  
  err = cudaMalloc((void **)&inputKangaroo,kangarooSize);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n",cudaGetErrorString(err));
    return;
  }
  
  err = cudaHostAlloc(&inputKangarooPinned,kangarooSize,cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }
  
  err = cudaMalloc((void **)&w_inputKangaroo,kangarooSize);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n",cudaGetErrorString(err));
    return;
  }
  
  err = cudaHostAlloc(&w_inputKangarooPinned,kangarooSize,cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if(err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n",cudaGetErrorString(err));
    return;
  }
  
  // OutputHash
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  
  // Jump array
  jumpSize = NB_JUMP * 8 * 4;
  err = cudaHostAlloc(&jumpPinned, jumpSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate jump pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  lostWarning = false;
  initialised = true;

}

GPUEngine::~GPUEngine() {

  cudaFree(inputKangaroo);// Tame points and keys
  cudaFreeHost(inputKangarooPinned);
  cudaFree(w_inputKangaroo);// Wild points and keys
  cudaFreeHost(w_inputKangarooPinned);
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);
  cudaFreeHost(jumpPinned);

}

int GPUEngine::GetMemory() {
  return kangarooSize + outputSize + jumpSize;
}

int GPUEngine::GetGroupSize() {
  return GPU_GRP_SIZE;
}

bool GPUEngine::GetGridSize(int gpuId,int *x,int *y) {

  if(*x <= 0 || *y <= 0) {

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if(error_id != cudaSuccess) {
      printf("GPUEngine: CudaGetDeviceCount %s\n",cudaGetErrorString(error_id));
      return false;
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if(deviceCount == 0) {
      printf("GPUEngine: There are no available device(s) that support CUDA\n");
      return false;
    }

    if(gpuId >= deviceCount) {
      printf("GPUEngine::GetGridSize() Invalid gpuId\n");
      return false;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,gpuId);

    if(*x <= 0) *x = 2 * deviceProp.multiProcessorCount;
    if(*y <= 0) *y = 2 * _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
	//if(*x <= 0) *x = deviceProp.multiProcessorCount;
    //if(*y <= 0) *y = _ConvertSMVer2Cores(deviceProp.major,deviceProp.minor);
    if(*y <= 0) *y = 128;

  }

  return true;

}

void *GPUEngine::AllocatePinnedMemory(size_t size) {

  void *buff;

  cudaError_t err = cudaHostAlloc(&buff,size,cudaHostAllocPortable);
  if(err != cudaSuccess) {
    printf("GPUEngine: AllocatePinnedMemory: %s\n",cudaGetErrorString(err));
    return NULL;
  }

  return buff;

}

void GPUEngine::FreePinnedMemory(void *buff) {
  cudaFreeHost(buff);
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
     NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i=0;i<deviceCount;i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem / 1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

bool GPUEngine::callKernel() {

  // Reset nbFound
  cudaMemset(outputPrefix,0,4);
  
  // Call the kernel 
  comp_keys << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
		(inputKangaroo, w_inputKangaroo, DPmodule, hop_modulo, maxFound, outputPrefix);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;

}


bool GPUEngine::SetKangaroos(Point *p, Point *wp, Int *d, Int *wd, Int *distance, Int *px, Int *py) {

  // Sets the kangaroos of each thread
  int gSize = 12 * GPU_GRP_SIZE;
  int strideSize = nbThreadPerGroup * 12;
  int nbBlock = nbThread / nbThreadPerGroup;
  int blockSize = nbThreadPerGroup * gSize;
  int idx = 0;

  for(int b = 0; b < nbBlock; b++) {
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
      for(int t = 0; t < nbThreadPerGroup; t++) {

        // X
        inputKangarooPinned[b * blockSize + g * strideSize + t + 0 * nbThreadPerGroup] = p[idx].x.bits64[0];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 1 * nbThreadPerGroup] = p[idx].x.bits64[1];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 2 * nbThreadPerGroup] = p[idx].x.bits64[2];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup] = p[idx].x.bits64[3];

        // Y
        inputKangarooPinned[b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup] = p[idx].y.bits64[0];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup] = p[idx].y.bits64[1];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 6 * nbThreadPerGroup] = p[idx].y.bits64[2];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 7 * nbThreadPerGroup] = p[idx].y.bits64[3];

        // Distance
        inputKangarooPinned[b * blockSize + g * strideSize + t + 8 * nbThreadPerGroup] = d[idx].bits64[0];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 9 * nbThreadPerGroup] = d[idx].bits64[1];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 10 * nbThreadPerGroup] = d[idx].bits64[2];
        inputKangarooPinned[b * blockSize + g * strideSize + t + 11 * nbThreadPerGroup] = d[idx].bits64[3];

        // Wild X
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 0 * nbThreadPerGroup] = wp[idx].x.bits64[0];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 1 * nbThreadPerGroup] = wp[idx].x.bits64[1];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 2 * nbThreadPerGroup] = wp[idx].x.bits64[2];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 3 * nbThreadPerGroup] = wp[idx].x.bits64[3];

        // Wild Y
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 4 * nbThreadPerGroup] = wp[idx].y.bits64[0];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 5 * nbThreadPerGroup] = wp[idx].y.bits64[1];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 6 * nbThreadPerGroup] = wp[idx].y.bits64[2];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 7 * nbThreadPerGroup] = wp[idx].y.bits64[3];

        // Wild Distance
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 8 * nbThreadPerGroup] = wd[idx].bits64[0];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 9 * nbThreadPerGroup] = wd[idx].bits64[1];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 10 * nbThreadPerGroup] = wd[idx].bits64[2];
        w_inputKangarooPinned[b * blockSize + g * strideSize + t + 11 * nbThreadPerGroup] = wd[idx].bits64[3];
		
		idx++;
      }
    }
  }

  // Fill device memory and free pinned mem
  cudaMemcpy(inputKangaroo,inputKangarooPinned,kangarooSize,cudaMemcpyHostToDevice);
  cudaMemcpy(w_inputKangaroo,w_inputKangarooPinned,kangarooSize,cudaMemcpyHostToDevice);
  
  if(!rekey) {
    // We do not need the input pinned memory anymore
	cudaFreeHost(inputKangarooPinned);
    inputKangarooPinned = NULL;
	
	cudaFreeHost(w_inputKangarooPinned);
    w_inputKangarooPinned = NULL;
  }
  
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: SetKangaroos: %s\n",cudaGetErrorString(err));
	return false;
  }
  
  // Set Jump
  for (int i=0;i< NB_JUMP;i++) {
	memcpy(jumpPinned + 4*i, distance[i].bits64, 32);	
  }
  cudaMemcpyToSymbol(dS, jumpPinned, jumpSize);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
	printf("GPUEngine: SetKangaroos: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
	return false;
  }

  for(int i = 0; i < NB_JUMP; i++) {
	memcpy(jumpPinned + 4 * i, px[i].bits64, 32);	
  }
  cudaMemcpyToSymbol(Spx, jumpPinned, jumpSize);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
	printf("GPUEngine: SetKangaroos: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
	return false;
  }

  for(int i = 0; i < NB_JUMP; i++) {
	memcpy(jumpPinned + 4 * i, py[i].bits64, 32);	
  }
  cudaMemcpyToSymbol(Spy, jumpPinned, jumpSize);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
	printf("GPUEngine: SetKangaroos: Failed to copy to constant memory: %s\n",cudaGetErrorString(err));
	return false;
  }
    
  return callKernel();

}

bool GPUEngine::callKernelAndWait() {

  // Debug function
  callKernel();
  cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess) {
    printf("GPUEngine: callKernelAndWait: %s\n",cudaGetErrorString(err));
    return false;
  }

  return true;

}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound, bool spinWait) {

  prefixFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputPrefixPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }
    
  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy(outputPrefixPinned, outputPrefix, nbFound*ITEM_SIZE + 4, cudaMemcpyDeviceToHost);
    
  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputPrefixPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;
	
	uint64_t *tpx = (uint64_t *)itemPtr;
	it.tpx.bits64[0] = tpx[0];
    it.tpx.bits64[1] = tpx[1];
    it.tpx.bits64[2] = tpx[2];
    it.tpx.bits64[3] = tpx[3];
    it.tpx.bits64[4] = 0;
	
	uint64_t *tkey = (uint64_t *)(itemPtr + 8);
    it.tkey.bits64[0] = tkey[0];
    it.tkey.bits64[1] = tkey[1];
    it.tkey.bits64[2] = tkey[2];
    it.tkey.bits64[3] = tkey[3];
    it.tkey.bits64[4] = 0;
	
	it.type = itemPtr[16];// 1 Tame 2 Wild
	it.thId = itemPtr[17];
	
	prefixFound.push_back(it);
  }

  return callKernel();

}
