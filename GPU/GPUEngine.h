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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"

// Number of GROUP SIZE
#define GPU_GRP_SIZE 32//64// Given the total time it takes to find a solution, not the speed of the search.
#define NB_SPIN 16
#define GPU_OUTPUT_CHECK 0// 1 Enable 0 Disable

#define NB_JUMP 32

#define ITEM_SIZE 72
#define ITEM_SIZE32 (ITEM_SIZE/4)


typedef struct {
  
  Int tpx;
  Int tkey;
  uint32_t type;// 1 Tame 2 Wild
  uint32_t thId;
  
} ITEM;

class GPUEngine {

public:

  GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey, uint32_t pow2w, uint64_t totalK, uint32_t hop_modulo, int power, int fixedDP); 
  ~GPUEngine();  
  bool SetKangaroos(Point *p, Point *wp, Int *d, Int *wd, Int *distance, Int *px, Int *py);  
  bool Launch(std::vector<ITEM> &prefixFound, bool spinWait=false);  
  int GetNbThread();
  int GetGroupSize();
  int GetMemory();
  static void GenerateCode(Secp256K1 *secp, int size);
  bool callKernelAndWait();
  bool callKernel();
  
  std::string deviceName;
  
  static void *AllocatePinnedMemory(size_t size);
  static void FreePinnedMemory(void *buff);
  static void PrintCudaInfo();
  static bool GetGridSize(int gpuId,int *x,int *y);
    
private:


  int nbThread;
  int nbThreadPerGroup;
  int kangarooSize;
  uint64_t *inputKangaroo;// Tame points and keys
  uint64_t *inputKangarooPinned;
  uint64_t *w_inputKangaroo;// Wild points and keys
  uint64_t *w_inputKangarooPinned;
  
  uint32_t *outputPrefix;
  uint32_t *outputPrefixPinned;
  uint64_t *jumpPinned;
  bool initialised;
  bool lostWarning;
  bool rekey;
  uint64_t DPmodule;
  uint32_t hop_modulo;
  uint32_t maxFound;
  uint32_t outputSize;
  uint32_t jumpSize;
  
};

#endif // GPUENGINEH
