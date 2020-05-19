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

#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif
#include <chrono>

#include <iostream>
#include <fstream>

using namespace std;

Point	Sp[256];
Int		dS[256];

// ----------------------------------------------------------------------------


// max=2^48, with fractional part
char * prefSI_double(char *s, size_t s_size, double doNum) {

	size_t ind_si = 0;
	char prefSI_list[9] = { ' ', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y' };

	while ((uint64_t)(doNum/1000) > 0) {
		ind_si += 1;
		doNum /= 1000;
		if (ind_si > 100) {
			printf("\n[FATAL_ERROR] infinity loop in prefSI!\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ind_si < sizeof(prefSI_list) / sizeof(prefSI_list[0])) {
		snprintf(&s[0], s_size, "%5.1lf", doNum);
		snprintf(&s[0+5], s_size-5, "%c", prefSI_list[ind_si]);
	}
	else {
		snprintf(&s[0], s_size, "infini");
	}

	return s;
}


// max=2^256, without fractional part
char * prefSI_Int(char *s, size_t s_size, Int bnNum) {

	size_t ind_si = 0;
	char prefSI_list[9] = { ' ', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y' };

	Int bnZero; bnZero.SetInt32(0);
	Int bn1000; bn1000.SetInt32(1000);
	Int bnTmp;	bnTmp = bnNum;
	bnTmp.Div(&bn1000);
	while (bnTmp.IsGreater(&bnZero)) {
		ind_si += 1;
		bnTmp.Div(&bn1000);
		bnNum.Div(&bn1000);
		if (ind_si > 100) {
			printf("\n[FATAL_ERROR] infinity loop in prefSI!\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ind_si < sizeof(prefSI_list) / sizeof(prefSI_list[0])) {
		snprintf(&s[0], s_size, "%3s.0", bnNum.GetBase10().c_str());
		snprintf(&s[0+5], s_size-5, "%c", prefSI_list[ind_si]);
	}
	else {
		snprintf(&s[0], s_size, "infini");
	}

	return s;
}


void passtime_tm(char *s, size_t s_size, const struct tm *tm) {

	size_t offset_start = 0;

	if (tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %1iy", tm->tm_year - 70);
		offset_start += 3;
		if (((tm->tm_year - 70) / 10) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 10000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 10000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000000000) != 0) offset_start += 1;
	}
	if (tm->tm_mon > 0 || tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %2im", tm->tm_mon);
		offset_start += 4;
	}
	if (tm->tm_mday - 1 > 0 || tm->tm_mon > 0 || tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %2id", tm->tm_mday - 1);
		offset_start += 4;
	}

	snprintf(&s[offset_start], s_size - offset_start, " %02i:%02i:%02is", tm->tm_hour, tm->tm_min, tm->tm_sec);
}


void passtime(char *s, size_t s_size, Int& bnSec, double usec = 0.0, char v[] = "11111100") {

	size_t ofst = 0;

	Int bnTmp;
	Int bnZero; bnZero.SetInt32(0);
	char buff_s6[6+1] = {0};

	Int Y_tmp; Y_tmp.SetInt32(0);
	Int M_tmp; M_tmp.SetInt32(0);
	Int d_tmp; d_tmp.SetInt32(0);
	Int h_tmp; h_tmp.SetInt32(0);
	Int m_tmp; m_tmp.SetInt32(0);
	Int s_tmp; s_tmp.SetInt32(0);

	int ms_tmp = 0, us_tmp = 0;
	if (usec != 0.0) {
		ms_tmp = (int)((uint64_t)(usec * 1000) % 1000);
		us_tmp = (int)((uint64_t)(usec * 1000000) % 1000);
	}

	if (v[0] != 48) {	// year
		Y_tmp = bnSec; 
		bnTmp.SetInt32(60*60*24*30*12); Y_tmp.Div(&bnTmp);
		if (Y_tmp.IsGreater(&bnZero)) {
			prefSI_Int(buff_s6, sizeof(buff_s6), Y_tmp);
			snprintf(&s[ofst], s_size-ofst, " %6sy", buff_s6); ofst += 8;
		}
	}
	if (v[1] != 48) { //month
		M_tmp = bnSec; 
		bnTmp.SetInt32(60*60*24*30); M_tmp.Div(&bnTmp); 
		bnTmp.SetInt32(12); M_tmp.Mod(&bnTmp);
		if (M_tmp.IsGreater(&bnZero) || Y_tmp.IsGreater(&bnZero)) {
			snprintf(&s[ofst], s_size-ofst, " %2sm", M_tmp.GetBase10().c_str()); ofst += 4;
		}
	}
	if (v[2] != 48) { //day
		d_tmp = bnSec;
		bnTmp.SetInt32(60*60*24); d_tmp.Div(&bnTmp);
		bnTmp.SetInt32(30); d_tmp.Mod(&bnTmp);
		if (d_tmp.IsGreater(&bnZero) || M_tmp.IsGreater(&bnZero) || Y_tmp.IsGreater(&bnZero)) {
			snprintf(&s[ofst], s_size-ofst, " %02sd", d_tmp.GetBase10().c_str()); ofst += 4;
		}
	}
	if (v[3] != 48) { //hour
		h_tmp = bnSec;
		bnTmp.SetInt32(60*60); h_tmp.Div(&bnTmp);
		bnTmp.SetInt32(24); h_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, " %02s", h_tmp.GetBase10().c_str()); ofst += 3;
		}
	}
	if (v[4] != 48) { //min
		m_tmp = bnSec;
		bnTmp.SetInt32(60); m_tmp.Div(&bnTmp);
		bnTmp.SetInt32(60); m_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, ":%02s", m_tmp.GetBase10().c_str()); ofst += 3;
		}
	}
	if (v[5] != 48) { //sec
		s_tmp = bnSec;
		bnTmp.SetInt32(60); s_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, ":%02s", s_tmp.GetBase10().c_str()); ofst += 3;
		}
	}


	if (v[6] != 48) { //msec
		if ((v[6] == 49) 
			|| (Y_tmp.IsEqual(&bnZero) && M_tmp.IsEqual(&bnZero) && d_tmp.IsEqual(&bnZero) 
				&& h_tmp.IsEqual(&bnZero) && m_tmp.IsEqual(&bnZero) && s_tmp.IsEqual(&bnZero)
				)
			) {
			snprintf(&s[ofst], s_size-ofst, " %03dms", ms_tmp); ofst += 6;
		}
	}

	if (v[7] != 48) { //usec
		if ((v[7] == 49)
			|| (Y_tmp.IsEqual(&bnZero) && M_tmp.IsEqual(&bnZero) && d_tmp.IsEqual(&bnZero)
				&& h_tmp.IsEqual(&bnZero) && m_tmp.IsEqual(&bnZero) && s_tmp.IsEqual(&bnZero)
				&& ms_tmp==0)
			) {
			snprintf(&s[ofst], s_size-ofst, " %03dus", us_tmp); ofst += 6;
		}
	}

}



// ----------------------------------------------------------------------------

VanitySearch::VanitySearch(Secp256K1 *secp, vector<Point> &targetPubKeys, Point targetPubKey, structW *stR, int nbThread, int KangPower, int DPm, bool stop, std::string outputFile, int flag_verbose, uint32_t maxFound, uint64_t rekey, bool flag_comparator) 
  :targetPubKeys(targetPubKeys) {

  this->secp = secp;

  this->targetPubKeys = targetPubKeys;
  this->targetPubKey = targetPubKey;

  this->stR = stR;
  this->bnL = stR->bnL;
  this->bnU = stR->bnU;
  this->bnW = stR->bnW;
  this->pow2L = stR->pow2L;
  this->pow2U = stR->pow2U;
  this->pow2W = stR->pow2W;
  this->bnM = stR->bnM;
  this->bnWsqrt = stR->bnWsqrt;
  this->pow2Wsqrt = stR->pow2Wsqrt;
  
  this->nbThread = nbThread;
  this->KangPower = KangPower;
  this->DPm = DPm;
  this->flag_comparator = flag_comparator;
  
  this->nbGPUThread = 0;
  this->maxFound = maxFound;
  this->rekey = rekey;
    
  this->outputFile = outputFile;
  this->flag_verbose = flag_verbose;
  
  lastRekey = 0;
  
  printf("[rangeW]	2^%u..2^%u ; W = U - L = 2^%u\n"
	  , pow2L, pow2U
	  , pow2W
  );

  /////////////////////////////////////////////////
  // hashtable for distinguished points

  // DPht,DTht,DWht - points, distinguished points of Tp/Wp
  // in hashtable, provides uniqueness distinguished points

  countDT = countDW = countColl = 0;
  //maxDP = 1 << 10; // 2^10=1024
  maxDP = 1 << 20; // 2^20=1048576
  printf("[DPsize]	%llu (hashtable size)\n", maxDP);

  HASH_SIZE = 2 * maxDP;

  DPht = (hashtb_entry *)calloc(HASH_SIZE, sizeof(hashtb_entry));

  if (NULL == DPht) {
	  printf("\n[FATAL ERROR] can't alloc mem %.2f %s \n", (float)(HASH_SIZE) * sizeof(hashtb_entry)/1024/1024/1024, "GiB");
	  exit(EXIT_FAILURE);
  }

  /////////////////////////////////////////////////
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  printf("[pubkey#%d] loaded\n", pow2U);
  printf("[Xcoordinate] %064s\n", targetPubKey.x.GetBase16().c_str());
  printf("[Ycoordinate] %064s\n", targetPubKey.y.GetBase16().c_str());

  if (!secp->EC(targetPubKey)) {
	  printf("\n[FATAL_ERROR] invalid public key (Not lie on elliptic curve)\n");
	  exit(EXIT_FAILURE);
  }

  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  /////////////////////////////////////////////////
  // pre-compute set S(i) jumps of pow2 jumpsize

  Sp[0] = secp->G;
  dS[0].SetInt32(1);
  for (int i = 1; i < 256; ++i) {
	  dS[i].Add(&dS[i-1], &dS[i-1]);
	  Sp[i] = secp->DoubleAffine(Sp[i-1]);
  }
  printf("[+] Sp-table of pow2 points - ready \n");
  
}

// ----------------------------------------------------------------------------

bool VanitySearch::output(string msg) {
  
  FILE *f = stdout;
  f = fopen(outputFile.c_str(), "a");
    
  if (f == NULL) {
	  printf("[error] Cannot open file '%s' for writing! \n", outputFile.c_str());
	  f = stdout;
	  return false;
  } 
  else {
	  fprintf(f, "%s\n", msg.c_str());
	  fclose(f);
	  return true;
  }
}

// ----------------------------------------------------------------------------

bool VanitySearch::outputgpu(string msg) {
  
  FILE *f = stdout;
  f = fopen(outputFile.c_str(), "a");
    
  if (f == NULL) {
	  printf("[error] Cannot open file '%s' for writing! \n", outputFile.c_str());
	  f = stdout;
	  return false;
  } 
  else {
	  fprintf(f, "\nPriv: %s\n", msg.c_str());
	  fprintf(f, "\nTips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9 \n");
	  fclose(f);
	  return true;
  }
}

// ----------------------------------------------------------------------------

bool VanitySearch::checkPrivKeyCPU(Int &checkPrvKey, Point &pSample) {

  Point checkPubKey = secp->ComputePubKey(&checkPrvKey);
	  
  if (!checkPubKey.equals(pSample)) {
	  if (flag_verbose > 1) {
		  printf("[pubkeyX#%d] %064s \n", pow2U, checkPubKey.x.GetBase16().c_str());
		  printf("[originX#%d] %064s \n", pow2U, pSample.x.GetBase16().c_str());

		  printf("[pubkeyY#%d] %064s \n", pow2U, checkPubKey.y.GetBase16().c_str());
		  printf("[originY#%d] %064s \n", pow2U, pSample.y.GetBase16().c_str());
	  }
	  return false;
  }
  return true;
}


// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKey(LPVOID lpParam) {
#else
void *_FindKey(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void *_FindKeyGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyGPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _SolverGPU(LPVOID lpParam) {
#else
void *_SolverGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolverGPU(p);
  return 0;
}

// ----------------------------------------------------------------------------



void VanitySearch::FindKeyCPU(TH_PARAM *ph) {

  int thId = ph->threadId;

  if (flag_verbose > 0)
	  printf("[th][%s#%d] run.. \n", (ph->type ? "wild" : "tame"), (ph->type ? thId+1-xU : thId+1));


  countj[thId] = 0;
  ph->hasStarted = true;


  while (!flag_endOfSearch) {

	  countj[thId] += 1;

	  // check, is it distinguished point ?
	  if (!(ph->Kp.x.bits64[0] % DPmodule)) {

		  //printf("[Xn0] 0x%016llx \n", ph->Kp.x.bits64[0]);
		  //printf("[Xcoord] %064s \n", ph->Kp.x.GetBase16().c_str());

		  // send new distinguished point to parent
		  #ifdef WIN64
		  WaitForSingleObject(ghMutex, INFINITE);
		  #else
		  pthread_mutex_lock(&ghMutex);
		  #endif

		  uint64_t entry = ph->Kp.x.bits64[0] & (HASH_SIZE-1);
		  while (DPht[entry].n0 != 0) {

			  if (DPht[entry].n0 == ph->Kp.x.bits64[0]
				  //&& DPht[entry].n1 == ph->Kp.x.bits64[1]
				  //&& DPht[entry].n2 == ph->Kp.x.bits64[2]
				  //&& DPht[entry].n3 == ph->Kp.x.bits64[3]
				  && !flag_endOfSearch
				  ) {

					  //printf("[X] 0x%064s\n", ph->Kp.x.GetBase16().c_str());

					  if (ph->dK.IsLower(&DPht[entry].distance))
						  resultPrvKey.Sub(&DPht[entry].distance, &ph->dK);
					  else if (ph->dK.IsGreater(&DPht[entry].distance))
						  resultPrvKey.Sub(&ph->dK, &DPht[entry].distance);
					  else {
						  printf("\n[FATAL_ERROR] dT == dW !!!\n");
						  //exit(EXIT_FAILURE);
					  }

					  if (checkPrivKeyCPU(resultPrvKey, targetPubKey)) {
						  flag_endOfSearch = true;
						  break;
					  }
					  else {
						  ++countColl;
						  printf("\n");
						  printf("[warn] hashtable collision(#%llu) found! \n", countColl);
						  //printf("[warn] hashtable collision(#%llu) found! 0x%016llX \n", countColl, ph->Kp.x.bits64[0]);
						  //printf("[warn] Xcoord=%064s \n", ph->Kp.x.GetBase16().c_str());
						  //printf("[warn] wrong prvkey 0x%064s \n", resultPrvKey.GetBase16().c_str());
					  }

			  }
			  entry = (entry + (ph->Kp.x.bits64[1] | 1)) & (HASH_SIZE-1);
		  }
		  //if (flag_endOfSearch) break;
		  
		  if (ph->type) { ++countDW; } else { ++countDT; }
		  if (flag_verbose > 1) {
			  printf("\n[th][%s#%d][DP %dT+%dW=%llu+%llu=%llu] new distinguished point!\n"
				  , (ph->type ? "wild" : "tame"), (ph->type ? thId+1 - xU : thId+1)
				  , xU, xV, countDT, countDW, countDT+countDW
			  );
		  }
		  if (countDT+countDW >= maxDP) {
			  printf("\n[FATAL_ERROR] DP hashtable overflow! %dT+%dW=%llu+%llu=%llu (max=%llu)\n"
				  , xU, xV, countDT, countDW, countDT+countDW, maxDP
			  );
			  exit(EXIT_FAILURE);
		  }

		  
		  DPht[entry].distance = ph->dK;
		  DPht[entry].n0 = ph->Kp.x.bits64[0];
		  //DPht[entry].n1 = ST.n[1];
		  //DPht[entry].n2 = ST.n[2];
		  //DPht[entry].n3 = ST.n[3];

		  #ifdef WIN64
		  ReleaseMutex(ghMutex);
		  #else
		  pthread_mutex_unlock(&ghMutex);
		  #endif
	  }
	  //if (flag_endOfSearch) break;

	  uint64_t pw = ph->Kp.x.bits64[0] % JmaxofSp;

	  //nowjumpsize = 1 << pw
	  Int nowjumpsize = dS[pw];

	  ph->dK.Add(&nowjumpsize);
	  
	  	  
	  // Jacobian points addition
	  //ph->Kp = secp->AddJacobian(ph->Kp, Sp[pw]); ph->Kp.Reduce(); 

	  // Affine points addition
	  ph->Kp = secp->AddAffine(ph->Kp, Sp[pw]);
  }

  ph->isRunning = false;

}

// ----------------------------------------------------------------------------

void VanitySearch::getGPUStartingKeys(int thId, int groupSize, int nbKangaroo, Int *keys, Point *p, uint64_t *n_count) {

  int tl = 0;
  int trbit = pow2W - tl;// pow2W-0 ?
  Int sk;
  sk.SetInt32(1);
  sk.Neg();
  //printf("\n GPU thId: %d ", thId);
  int skeybit = bnL.GetBitLength();
  printf("GPU Bits: %d ", skeybit);
  printf("\nGPU Tame Points: [M] + Rand(pow2W-%d) ", tl);
  key2 = bnL;
  key3 = bnU;
  key3.Add(&sk);
  for (int i = 0; i < nbKangaroo; i++) {
	// Get Tame keys
	Int offT((uint64_t)i);
	if (rekey > 1) {
		/*
		keys[i].SetInt32(0);
		keys[i].Rand(skeybit);
		while(strcmp(keys[i].GetBase16().c_str(), key2.GetBase16().c_str()) < 0 || strcmp(keys[i].GetBase16().c_str(), key3.GetBase16().c_str()) > 0){
			//printf("\n GPU Random Key: %s < bnL or > bnU New Random true \n", keys[i].GetBase16().c_str());
			wkey.Rand(skeybit);
			keys[i] = wkey;
		}
		*/		
		keys[i] = bnM;
		if (i > 0) {
			wkey.Rand(trbit);
			keys[i].Add(&wkey);
			keys[i].Add(&offT);// if bad Rand ?
		}
		// Check GPU Random Key
		if (i < 10) printf("\nGPU Tame Starting Key %d: %s ", i, keys[i].GetBase16().c_str());
		if (i == nbKangaroo-1) printf("\nGPU Tame Starting Key %d: %s Kangaroo: %d \n", i, keys[i].GetBase16().c_str(), nbKangaroo);
		
	} else {
      // Get Tame keys
	  keys[i] = bnM;
	  if (i > 0) {
		wkey.Rand(trbit);
		keys[i].Add(&wkey);
		keys[i].Add(&offT);// if bad Rand ?
	  }
	  // Check GPU Random Key
	  if (i < 10) printf("\nGPU Tame Starting Key %d: %s ", i, keys[i].GetBase16().c_str());
	  if (i == nbKangaroo-1) printf("\nGPU Tame Starting Key %d: %s Kangaroo: %d \n", i, keys[i].GetBase16().c_str(), nbKangaroo);
	  
	}
    // For check GPU code
	//keys[i].SetBase16("FF");
	//keys[i].SetInt32(10);
	
	//Int k(keys + i); the error ?
	Int k(keys[i]);
    p[i] = secp->ComputePublicKey(&k);
  }
  ++ *n_count;
}

// ----------------------------------------------------------------------------

void VanitySearch::getGPUStartingWKeys(int thId, int groupSize, int nbKangaroo, Point w_targetPubKey, Int *w_keys, Point *w_p) {

  int wl = 0;// set optimal space
  int wrbit = pow2W - wl;// pow2W-1 ?
  printf("\nGPU Wild Points: [Target] + Rand(pow2W-%d) ", wl);
  for (int i = 0; i < nbKangaroo; i++) {
	// Get Wild keys
	Int woffT((uint64_t)i);
	w_keys[i].SetInt32(0);
	wkey.Rand(wrbit);
	w_keys[i].Add(&wkey);
	w_keys[i].Add(&woffT);// if bad Rand
	//
	Int wk(w_keys[i]);
	w_p[i] = secp->ComputePublicKey(&wk);
	if (1) { // Set 0 for check compute points in GPU
		w_p[i] = secp->AddAffine(w_targetPubKey, w_p[i]);
	}
	if (i < 10) printf("\nGPU Wild Starting Key %d: %s ", i, w_keys[i].GetBase16().c_str());
	if (i == nbKangaroo-1) printf("\nGPU Wild Starting Key %d: %s Kangaroo: %d \n", i, w_keys[i].GetBase16().c_str(), nbKangaroo);
	  
  }
  
}

// ----------------------------------------------------------------------------
/*
void VanitySearch::CreateJumpTable(uint32_t Jmax, int pow2w) {
	
	int jumpBit = pow2w / 2 + 1;
	int jumpBit_max = (int)Jmax;
	printf("Create Jump Table Max Jump: %d \n", jumpBit_max);
	double maxAvg = pow(2.0,(double)jumpBit - 0.95);
	double minAvg = pow(2.0,(double)jumpBit - 1.05);
	double distAvg;
	printf("Jump Avg distance min: 2^%.2f\n",log2(minAvg));
	printf("Jump Avg distance max: 2^%.2f\n",log2(maxAvg));
	
	Int totalDist;
	totalDist.SetInt32(0);
	// Original code
	//for (int i = 0; i < NB_JUMP; i++) {
		//jumpDistance[i].Rand(jumpBit);
		//totalDist.Add(&jumpDistance[i]);
		//Point J = secp->ComputePublicKey(&jumpDistance[i]);
		//jumpPointx[i].Set(&J.x);
		//jumpPointy[i].Set(&J.y);		
	//}
	
	// Add small jumps
	int small_jump = NB_JUMP / 2;// #define NB_JUMP 32
	Point *Sp = new Point[small_jump];
	Int *dS = new Int[small_jump];
	Sp[0] = secp->G;
	dS[0].SetInt32(1);	
	jumpPointx[0].Set(&Sp[0].x);
	jumpPointy[0].Set(&Sp[0].y);
	jumpDistance[0].SetInt32(1);
	if (flag_verbose > 1) {
		printf("Jump: 0 Distance: %s \n", jumpDistance[0].GetBase16().c_str());
	}
	for (int i = 1; i < NB_JUMP; i++) {
		
		if (i < small_jump) {
			// Set small jump
			dS[i].Add(&dS[i-1], &dS[i-1]);
			Sp[i] = secp->DoubleAffine(Sp[i-1]);
			jumpDistance[i].Set(&dS[i]);
			totalDist.Add(&jumpDistance[i]);
			jumpPointx[i].Set(&Sp[i].x);
			jumpPointy[i].Set(&Sp[i].y);		
		}		
		else {
			// Set original jumps
			jumpDistance[i].Rand(jumpBit_max);
			totalDist.Add(&jumpDistance[i]);
			Point J = secp->ComputePublicKey(&jumpDistance[i]);
			jumpPointx[i].Set(&J.x);
			jumpPointy[i].Set(&J.y);
		}
		if (flag_verbose > 1) {
			printf("Jump: %d Distance: %s \n", i, jumpDistance[i].GetBase16().c_str());
		}
	}
	
	distAvg = totalDist.ToDouble() / (double)NB_JUMP;	
	
	printf("Jump Avg distance: 2^%.2f \n", log2(distAvg));	
	
}
*/

void VanitySearch::CreateJumpTable() {
	
	printf("[i] Create Jump Table Size: %d \n", (int)NB_JUMP);
	
	// Create Jump Table
	Point *Sp = new Point[NB_JUMP];// #define NB_JUMP 64// Do not change!
	Int *dS = new Int[NB_JUMP];
	Sp[0] = secp->G;
	dS[0].SetInt32(1);
	jumpPointx[0].Set(&Sp[0].x);
	jumpPointy[0].Set(&Sp[0].y);
	jumpDistance[0].SetInt32(1);
	if (flag_verbose > 1) {
		printf("Jump: 0 Distance: %s \n", jumpDistance[0].GetBase16().c_str());
	}
	for (int i = 1; i < NB_JUMP; i++) {
		
		dS[i].Add(&dS[i-1], &dS[i-1]);
		Sp[i] = secp->DoubleAffine(Sp[i-1]);
		jumpDistance[i].Set(&dS[i]);
		jumpPointx[i].Set(&Sp[i].x);
		jumpPointy[i].Set(&Sp[i].y);
		
		if (flag_verbose > 1) {
			printf("Jump: %d Distance: %s \n", i, jumpDistance[i].GetBase16().c_str());
		}
	}
}


// ----------------------------------------------------------------------------

bool VanitySearch::File2save(Int px,  Int key, int stype) {
	
	string str_px = px.GetBase16().c_str();
	string str_key = key.GetBase16().c_str();
	string getpx = "";
	string getkey = "";
	string p0 = "0";
	string k0 = "0";
	
	for (int i = (int)str_px.size(); i < 64; i++) {
		getpx.append(p0);
	}
	getpx.append(str_px);
	
	for (int i = (int)str_key.size(); i < 64; i++) {
		getkey.append(k0);
	}
	getkey.append(str_key);
	
	string str_write = getpx + " " + getkey;
	
	int lenstr = (int)str_write.length();
	
	//printf("\n lenstr: %d stype: %d \n", lenstr, stype);
	//printf("\n str_write: %s \n", str_write.c_str());
	
	// Write files 
	if (stype == 1 && lenstr == 129) {
		FILE *f1 = fopen("tame-1.txt", "a");
		if (f1 == NULL) {
			printf("\n[error] Cannot open file tame-1.txt for writing! %s \n", strerror(errno));
			f1 = stdout;
			return false;
		} else {
			fprintf(f1, "%s\n", str_write.c_str());
			fclose(f1);
			return true;
		}		
	}
	if (stype == 2 && lenstr == 129) {
		FILE *f2 = fopen("wild-1.txt", "a");
		if (f2 == NULL) {
			printf("\n[error] Cannot open file wild-1.txt for writing! %s \n", strerror(errno));
			f2 = stdout;
			return false;
		} else {
			fprintf(f2, "%s\n", str_write.c_str());
			fclose(f2);
			return true;
		}		
	}	
}

// ----------------------------------------------------------------------------

bool VanitySearch::File2save2(Int px,  Int key, int stype) {
	
	string str_px = px.GetBase16().c_str();
	string str_key = key.GetBase16().c_str();
	string getpx = "";
	string getkey = "";
	string p0 = "0";
	string k0 = "0";
	
	for (int i = (int)str_px.size(); i < 64; i++) {
		getpx.append(p0);
	}
	getpx.append(str_px);
	
	for (int i = (int)str_key.size(); i < 64; i++) {
		getkey.append(k0);
	}
	getkey.append(str_key);
	
	string str_write = getpx + " " + getkey;
	
	int lenstr = (int)str_write.length();
	
	//printf("\n lenstr: %d stype: %d \n", lenstr, stype);
	//printf("\n str_write: %s \n", str_write.c_str());
	
	// Write files 
	if (stype == 1 && lenstr == 129) {
		FILE *f1 = fopen("tame-2.txt", "a");
		if (f1 == NULL) {
			printf("\n[error] Cannot open file tame-2.txt for writing! %s \n", strerror(errno));
			f1 = stdout;
			return false;
		} else {
			fprintf(f1, "%s\n", str_write.c_str());
			fclose(f1);
			return true;
		}		
	}
	if (stype == 2 && lenstr == 129) {
		FILE *f2 = fopen("wild-2.txt", "a");
		if (f2 == NULL) {
			printf("\n[error] Cannot open file wild-2.txt for writing! %s \n", strerror(errno));
			f2 = stdout;
			return false;
		} else {
			fprintf(f2, "%s\n", str_write.c_str());
			fclose(f2);
			return true;
		}		
	}	
}

// ----------------------------------------------------------------------------

bool VanitySearch::Comparator() {
	
	// Used chrono for get compare time 
	auto begin = std::chrono::steady_clock::now();
	
	string WfileName = "wild.txt";
	string TfileName = "tame.txt";
	vector<string> Wpoint;
	vector<string> Wkey;

	vector<string> Tpoint;
	vector<string> Tkey;

	// Wild
	// Get file size wild
	FILE *fw = fopen(WfileName.c_str(), "rb");
	if (fw == NULL) {
		printf("Error: Cannot open %s %s\n", WfileName.c_str(), strerror(errno));
	}
	fseek(fw, 0L, SEEK_END);
	size_t wsz = ftell(fw); // Get bytes
	size_t nbWild = wsz / 129; // Get lines
	fclose(fw);
	// For check
	//printf("File wild.txt: %llu bytes %llu lines \n", (uint64_t)wsz, (uint64_t)nbWild);

	// Parse File Wild
	int nbWLine = 0;
	string Wline = "";
	ifstream inFileW(WfileName);
	Wpoint.reserve(nbWild);
	Wkey.reserve(nbWild);
	while (getline(inFileW, Wline)) {

		// Remove ending \r\n
		int l = (int)Wline.length() - 1;
		while (l >= 0 && isspace(Wline.at(l))) {
			Wline.pop_back();
			l--;
		}

		if (Wline.length() == 129) {
			Wpoint.push_back(Wline.substr(0, 64));
			Wkey.push_back(Wline.substr(65, 129));
			nbWLine++;
			// For check
			//printf(" %s    %d \n", Wpoint[0].c_str(), nbWLine);
			//printf(" %s    %d \n", Wkey[0].c_str(), nbWLine);						
		}
	}

	// Tame
	// Get file size tame
	FILE *ft = fopen(TfileName.c_str(), "rb");
	if (ft == NULL) {
		printf("Error: Cannot open %s %s\n", TfileName.c_str(), strerror(errno));
	}
	fseek(ft, 0L, SEEK_END);
	size_t tsz = ftell(ft); // Get bytes
	size_t nbTame = tsz / 129; // Get lines
	fclose(ft);
	// For check
	//printf("File tame.txt: %llu bytes %llu lines \n", (uint64_t)tsz, (uint64_t)nbTame);

	// Parse File Tame
	int nbTLine = 0;
	string Tline = "";
	ifstream inFileT(TfileName);
	Tpoint.reserve(nbTame);
	Tkey.reserve(nbTame);
	while (getline(inFileT, Tline)) {

		// Remove ending \r\n
		int l = (int)Tline.length() - 1;
		while (l >= 0 && isspace(Tline.at(l))) {
			Tline.pop_back();
			l--;
		}

		if (Tline.length() == 129) {
			Tpoint.push_back(Tline.substr(0, 64));
			Tkey.push_back(Tline.substr(65, 129));
			nbTLine++;
			// For check
			//printf(" %s    %d \n", Tpoint[0].c_str(), nbTLine);
			//printf(" %s    %d \n", Tkey[0].c_str(), nbTLine);
		}
	}

	// Compare lines
	int result = 0;
	string WDistance = "";
	string TDistance = "";
	for (int wi = 0; wi < nbWLine; wi++) {

		for (int ti = 0; ti < nbTLine; ti++) {
			
			if (strcmp(Wpoint[wi].c_str(), Tpoint[ti].c_str()) == 0) {
				result++;
				if (result > 0) { 
				printf("\n%d Compared lines Tame %d = Wild %d ", result, ti+1, wi+1);
				printf("\nTame Distance: 0x%s ", Tkey[ti].c_str());
				printf("\nWild Distance: 0x%s ", Wkey[wi].c_str());				
				}
				WDistance = Wkey[wi].c_str();
				TDistance = Tkey[ti].c_str();								
			}
		}

	}
	
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
	if (flag_verbose > 0) { 
		printf("\n[i] Comparator time: %.*f msec %s %llu bytes %s %llu bytes \n", 3, elapsed_ms, WfileName.c_str(), (uint64_t)wsz, TfileName.c_str(), (uint64_t)tsz); 
	}
	
	if (result > 0) {
		
		// Get SOLVED
		Int WDist;
		Int TDist;
		Int Priv;
		char *wd = new char [WDistance.length()+1];
		char *td = new char [TDistance.length()+1];
		strcpy(wd, WDistance.c_str());
		strcpy(td, TDistance.c_str());
		WDist.SetBase16(wd);
		TDist.SetBase16(td);
		Priv.SetInt32(0);
		
		if (TDist.IsLower(&WDist))
			Priv.Sub(&WDist, &TDist);
		else if (TDist.IsGreater(&WDist))
			Priv.Sub(&TDist, &WDist);
		else {
			printf("\n[FATAL_ERROR] Wild Distance == Tame Distance !!!\n");			
		}
		
		printf("\nSOLVED: 0x%s \n", Priv.GetBase16().c_str());
		printf("Tame Distance: 0x%s \n", TDist.GetBase16().c_str());
		printf("Wild Distance: 0x%s \n", WDist.GetBase16().c_str());
		printf("Tips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9  ;-) \n");
		printf("\n[i] Comparator time: %.*f msec %s %llu bytes %s %llu bytes \n", 3, elapsed_ms, WfileName.c_str(), (uint64_t)wsz, TfileName.c_str(), (uint64_t)tsz); 
		
		// SAVE SOLVED
		bool saved = outputgpu(Priv.GetBase16().c_str());
		if (saved) {
			printf("[i] Success saved to file %s\n", outputFile.c_str());
		}		
		return true;
	}
	return false;
}

// ----------------------------------------------------------------------------
/*
bool VanitySearch::TWSaveToDrive() {
	
	#ifdef WIN64
	#else
	// Copy tame.txt and wild.txt in drive
	string save_tame = "cp ./tame.txt ../drive/'My Drive'/tame.txt";
	string save_wild = "cp ./wild.txt ../drive/'My Drive'/wild.txt";
	const char* tsave = save_tame.c_str();
	const char* wsave = save_wild.c_str();
	bool tsaved = system(tsave);
	Timer::SleepMillis(1000);
	bool wsaved = system(wsave);
	if (!tsaved && !wsaved) {
		return true;
	}	
	#endif 
	return false;
}
*/
// ----------------------------------------------------------------------------
/*
bool VanitySearch::TWUpload() {
	
	#ifdef WIN64
	#else
	// Upload tame.txt and wild.txt 
	string upload_tame = "cp ../drive/'My Drive'/tame.txt ./tame.txt";
	string upload_wild = "cp ../drive/'My Drive'/wild.txt ./wild.txt";
	const char* tupload = upload_tame.c_str();
	const char* wupload = upload_wild.c_str();
	bool tup = system(tupload);
	Timer::SleepMillis(1000);
	bool wup = system(wupload);
	if (!tup && !wup) {
		return true;
	}	
	#endif 
	return false;
	
}
*/
// ----------------------------------------------------------------------------

void VanitySearch::FindKeyGPU(TH_PARAM *ph) {

  bool ok = true;

#ifdef WITHGPU

  // mean jumpsize
  // by Pollard ".. The best choice of m (mean jump size) is w^(1/2)/2 .."
  Int GPUmidJsize;
  //GPUmidJsize.SetInt32(0);
  GPUmidJsize.Set(&bnWsqrt);// GPUmidJsize = bnWsqrt;
  GPUmidJsize.ShiftR(1);// Wsqrt / 2
  GPUJmaxofSp = (uint32_t)getJmaxofSp(GPUmidJsize, dS);  
  
  printf("===========================================================================\n");
  
  GPUJmaxofSp += (uint32_t)KangPower;// + Kangaroo Power  
  printf("[i] Used Jump Table: G,2G,4G,8G,16G,...,(2^NB_JUMP)G\n");
  
  CreateJumpTable();
  
  printf("===========================================================================\n");
  
  //CreateJumpTable(GPUJmaxofSp, pow2W);
  
  // Global init
  int thId = ph->threadId;
  GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, (rekey!=0), pow2W, totalRW*2, GPUJmaxofSp, KangPower, DPm);
  // Create Kangaroos
  int nbThread = g.GetNbThread();
  int nbKang = nbThread * GPU_GRP_SIZE;
  Point *p = new Point[nbKang];// Tame point
  Point *wp = new Point[nbKang];// Wild point
  Int *keys = new Int[nbKang];// Tame start keys
  Int *wkeys = new Int[nbKang];// Wild start keys
  vector<ITEM> found;

  printf("GPU %s (%.1f MB used) \n",g.deviceName.c_str(), g.GetMemory() / 1048576.0);
    
  countj[thId] = 0;
  kadd_count = 0;
  uint64_t n_count = 0;
  bool f2save = false;
  bool f2save2 = false;
  
  // Get Tame points
  getGPUStartingKeys(thId, g.GetGroupSize(), nbKang, keys, p, &n_count);
  
  // Get Wild points and keys
  getGPUStartingWKeys(thId, g.GetGroupSize(), nbKang, targetPubKey, wkeys, wp);
  
  ok = g.SetKangaroos(p, wp, keys, wkeys, jumpDistance, jumpPointx, jumpPointy);
  
  ph->rekeyRequest = false;  
  ph->hasStarted = true;
  
  // GPU Thread
  
  // ReWrite/Create files ?
  
  ReWriteFiles();
  
  /*
  TWRevers = false;// Aggregation of distinguished points, if true - Set int pow2dp = 22;// Fixed in GPUEngine.cu
  
  if (TWRevers) { 
	bool F2Upload = TWUpload();
	if (F2Upload) {
		printf("\n[i] Upload tame.txt and wild.txt ");
	}
  }
  else {
	ReWriteFiles();
  }
  */
    
  // Solver chmod
  #ifdef WIN64
  #else
  bool setChmod = SolverChmod();
  #endif  
  
  // Select comparator type
  flag_comparator = true;
  if (flag_comparator) {
	printf("\n[i] Used Comparator in Python");
  } else {
	printf("\n[i] Used Comparator in C++");
  }
  
  flag_startGPU = true;  
  Point chp;
  uint32_t file_cnt;
  
  while (ok && !flag_endOfSearch) {

    if (ph->rekeyRequest) {
      getGPUStartingKeys(thId, g.GetGroupSize(), nbKang, keys, p, &n_count);
	  getGPUStartingWKeys(thId, g.GetGroupSize(), nbKang, targetPubKey, wkeys, wp);
      ok = g.SetKangaroos(p, wp, keys, wkeys, jumpDistance, jumpPointx, jumpPointy);
      ph->rekeyRequest = false;
	  kadd_count = 0;
    }
	
	// Call kernel
    ok = g.Launch(found);
	
	//printf("\n nbFound %d \n", (int)found.size());
	
	LOCK(ghMutex);
		
	for(int i=0;i<(int)found.size() && !flag_endOfSearch;++i) {
		
		ITEM it = found[i];
		
		Int Tpx(&it.tpx);
		Int Tkey(&it.tkey);
		
		if (it.type == 1 || it.type == 2) {
			
			//printf("\n it.thId: %08u \n", it.thId);			
			//printf("\n     Tpx: %s ", Tpx.GetBase16().c_str());
			//printf("\n    Tkey: %s ", Tkey.GetBase16().c_str());
			//printf("\n    type: %d \n", (int)it.type);// type 2 - Wild, 1 - Tame
			
			// Check Output Data GPU and Compute Tame keys
			if (it.type == 1 && GPU_OUTPUT_CHECK == 1) {
				
				Int chk(&Tkey);
				chp = secp->ComputePubKey(&chk);
				if (strcmp(Tpx.GetBase16().c_str(), chp.x.GetBase16().c_str()) != 0) {
					printf("\n[error] Check Output Data GPU and Compute keys, thId: %08u", it.thId);
					printf("\n Check  key: %s ", Tkey.GetBase16().c_str());
					printf("\n    Point x: %s ", chp.x.GetBase16().c_str());
					printf("\n DP Point x: %s \n", Tpx.GetBase16().c_str());
					printf("[i] Set GPU_GRP_SIZE 32 or 16 \n");
				}
			}
			
			file_cnt = kadd_count + (uint32_t)found.size();
			
			if (file_cnt % 2 == 0) {
				f2save = File2save(&Tpx, &Tkey, (int)it.type);
				if (!f2save) {
					printf("\n[error] Not File2save type: %d \n", (int)it.type);
				}
			} else {
				f2save2 = File2save2(&Tpx, &Tkey, (int)it.type);
				if (!f2save2) {
					printf("\n[error] Not File2save2 type: %d \n", (int)it.type);
				}
			}
			
		}
	}
	
	UNLOCK(ghMutex);
	
	//printf("\n kadd_count: %d ", kadd_count);
	
	if (ok) {
      ++ kadd_count;
	  countj[thId] += 2ULL * NB_SPIN * nbKang; // 2ULL Tame + Wild
    }

  }
    
  delete[] keys;
  delete[] p;
  delete[] wkeys;
  delete[] wp;

#else
  ph->hasStarted = true;
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

  ph->isRunning = false;
  
}

// ----------------------------------------------------------------------------

void VanitySearch::SolverGPU(TH_PARAM *ph) {
	
	// Wait Started GPU threads
	while (!flag_startGPU) {
		Timer::SleepMillis(500);
	}
	
	int slp = pow2Wsqrt * 1000;
	//slp *= 2;
	slp = 1000 * 10;
	
	printf("\n[+] Runing Comparator every: %d sec\n", slp / 1000);
		
	while (!flag_endOfSearch) {
		
		// Used c++ Comparator if DPmodule > 2^25 ?
		/*
		if (!flag_comparator) {
		
		bool solver = Comparator();// Low Speed Comparator ?
		
		if (solver) {
			#ifdef WIN64
			#else
			// Save result to drive
			string comm_save = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
			const char* csave = comm_save.c_str();
			bool saved = system(csave);
			#endif
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("[i] No Cleaning wild.txt and tame.txt \n");
			slp = 500;// ReWriteFiles
		}
		}
		*/
		//else {
		// Python Comparator
		// Get compare time
		auto begin = std::chrono::steady_clock::now();
		#ifdef WIN64
		// win solver-1
		string comm_cmp_win1 = "solver-1.py";
		const char* scmpw1 = comm_cmp_win1.c_str();
		bool solver_win1 = system(scmpw1);
		if (!solver_win1) {
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// win solver-2
		string comm_cmp_win2 = "solver-2.py";
		const char* scmpw2 = comm_cmp_win2.c_str();
		bool solver_win2 = system(scmpw2);
		if (!solver_win2) {
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// win solver-3
		string comm_cmp_win3 = "solver-3.py";
		const char* scmpw3 = comm_cmp_win3.c_str();
		bool solver_win3 = system(scmpw3);
		if (!solver_win3) {
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// win solver-4
		string comm_cmp_win4 = "solver-4.py";
		const char* scmpw4 = comm_cmp_win4.c_str();
		bool solver_win4 = system(scmpw4);
		if (!solver_win4) {
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		#else
		// solver-1
		string comm_cmp1 = "./solver-1.py";
		const char* scmp1 = comm_cmp1.c_str();		
		bool solver1 = system(scmp1);		
		if (!solver1) {
			// Copy result in drive
			string comm_save1 = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
			const char* csave1 = comm_save1.c_str();
			bool saved1 = system(csave1);
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// solver-2
		string comm_cmp2 = "./solver-2.py";
		const char* scmp2 = comm_cmp2.c_str();		
		bool solver2 = system(scmp2);		
		if (!solver2) {
			// Copy result in drive
			string comm_save2 = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
			const char* csave2 = comm_save2.c_str();
			bool saved2 = system(csave2);
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// solver-3
		string comm_cmp3 = "./solver-3.py";
		const char* scmp3 = comm_cmp3.c_str();		
		bool solver3 = system(scmp3);		
		if (!solver3) {
			// Copy result in drive
			string comm_save3 = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
			const char* csave3 = comm_save3.c_str();
			bool saved3 = system(csave3);
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		// solver-4
		string comm_cmp4 = "./solver-4.py";
		const char* scmp4 = comm_cmp4.c_str();		
		bool solver4 = system(scmp4);		
		if (!solver4) {
			// Copy result in drive
			string comm_save4 = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
			const char* csave4 = comm_save4.c_str();
			bool saved4 = system(csave4);
			flag_endOfSearch = true;
			//ReWriteFiles();
			printf("\n[i] No Cleaning wild-1.txt, wild-2.txt, tame-1.txt, tame-2.txt \n");
			slp = 500;// ReWriteFiles
		}
		#endif 
		auto end = std::chrono::steady_clock::now();
		//auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
		if (flag_verbose > 0) { printf("\nPython Comparator time: %.*f msec \n", 3, elapsed_ms); }
		//}
		/*
		if (TWRevers) {
			bool TWSave = TWSaveToDrive();
			if (TWSave) {
				printf("\n[i] Copy tame.txt and wild.txt in My Drive \n");
			}
		}
		*/
		Timer::SleepMillis(slp);		
	}	
}

// ----------------------------------------------------------------------------

bool VanitySearch::isAlive(TH_PARAM *p) {

  bool isAlive = true;
  int total = nbCPUThread + nbGPUThread;
  for(int i = 0 ; i < total ; ++i)
    isAlive = isAlive && p[i].isRunning;

  return isAlive;

}

// ----------------------------------------------------------------------------

bool VanitySearch::hasStarted(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; ++i)
    hasStarted = hasStarted && p[i].hasStarted;

  return hasStarted;

}

// ----------------------------------------------------------------------------

void VanitySearch::rekeyRequest(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
  p[i].rekeyRequest = true;

}

// ----------------------------------------------------------------------------

uint64_t VanitySearch::getGPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbGPUThread;i++)
    count += countj[0x80L+i];
  return count;

}


// ----------------------------------------------------------------------------

uint64_t VanitySearch::getCountJ() {

  uint64_t count = 0;
  for(int i = 0 ; i < nbCPUThread ; ++i)
    count += countj[i];
  return count;

}

// ----------------------------------------------------------------------------

uint64_t VanitySearch::getJmaxofSp(Int& optimalmeanjumpsize, Int * dS) {

	if (flag_verbose > 0) {
		printf("[optimal_mean_jumpsize] %s \n", optimalmeanjumpsize.GetBase16().c_str());
	}
		
	Int sumjumpsize; 
	sumjumpsize.SetInt32(0);

	Int now_meanjumpsize, next_meanjumpsize;
	Int Isub1, Isub2;

	Int Ii;
	for (int i = 1; i < 256; ++i) {

		Ii.SetInt32(i);

		//sumjumpsize  = (2**i)-1
		//sumjumpsize += 2**(i-1)
		//sumjumpsize += dS[i-1]
		sumjumpsize.Add(&dS[i-1]);

		//now_meanjumpsize = int(round(1.0*(sumjumpsize) / (i)));
		now_meanjumpsize = sumjumpsize; now_meanjumpsize.Div(&Ii);

		//next_meanjumpsize = int(round(1.0*(sumjumpsize + 2**i) / (i+1)));
		//next_meanjumpsize = int(round(1.0*(sumjumpsize + dS[i]) / (i+1)));
		next_meanjumpsize = sumjumpsize; next_meanjumpsize.Add(&dS[i]); 
		Ii.SetInt32(i+1); next_meanjumpsize.Div(&Ii); Ii.SetInt32(i);

		//if  optimalmeanjumpsize - now_meanjumpsize <= next_meanjumpsize - optimalmeanjumpsize : 
		Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize);
		Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize);

		if (flag_verbose > 1)
			printf("[meanjumpsize#%d] %s(now) <= %s(optimal) <= %s(next)\n", i
				, now_meanjumpsize.GetBase16().c_str()
				, optimalmeanjumpsize.GetBase16().c_str()
				, next_meanjumpsize.GetBase16().c_str()
			);

		//if (Isub1.IsLowerOrEqual(&Isub2)) invalid compare for signed int!!! only unsigned int correct compared.
		if (
			   ((Isub1.IsNegative() || Isub1.IsZero()) && Isub2.IsPositive())
			|| ( Isub1.IsNegative() && (Isub2.IsZero() || Isub2.IsPositive()))
			|| ((Isub1.IsPositive() || Isub1.IsZero()) && (Isub2.IsPositive() || Isub2.IsZero()) && Isub1.IsLowerOrEqual(&Isub2))
			|| ((Isub1.IsNegative() || Isub1.IsZero()) && (Isub2.IsNegative() || Isub2.IsZero()) && Isub1.IsLowerOrEqual(&Isub2))
		) {

			if (flag_verbose > 0)
				printf("[meanjumpsize#%d] %s(now) <= %s(optimal) <= %s(next)\n", i
					, now_meanjumpsize.GetBase16().c_str()
					, optimalmeanjumpsize.GetBase16().c_str()
					, next_meanjumpsize.GetBase16().c_str()
				);

			// location in keyspace on the strip
			if (flag_verbose > 0) {

				//if (optimalmeanjumpsize - now_meanjumpsize) >= 0:
				if (Isub1.IsZero() || Isub1.IsPositive()) {
					//len100perc = 60
					Int len100perc; len100perc.SetInt32(60);
					//size1perc = (next_meanjumpsize-now_meanjumpsize)//len100perc
					Int size1perc; size1perc.Sub(&next_meanjumpsize, &now_meanjumpsize);
					size1perc.Div(&len100perc);
					printf("[i] Sp[%d]|", i);
						//, '-'*(abs(optimalmeanjumpsize - now_meanjumpsize)//size1perc)
					Isub1.Abs(); Isub1.Div(&size1perc);
					for (uint32_t j = 0 ; j < Isub1.GetInt32() ; ++j) printf("-");
					printf("J");
						//, '-'*(abs(next_meanjumpsize - optimalmeanjumpsize)//size1perc)
					Isub2.Abs(); Isub2.Div(&size1perc);
					for (uint32_t j = 0 ; j < Isub2.GetInt32() ; ++j) printf("-");
					printf("|Sp[%d]\n", i+1);
					//if (1.0 * abs(optimalmeanjumpsize - now_meanjumpsize) / abs(next_meanjumpsize - optimalmeanjumpsize) >= 0.25) {
					Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize); Isub1.Abs();
					Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize); Isub2.Abs();
					if ((float) Isub1.GetInt32() / Isub2.GetInt32() >= 0.25) {
						printf("[i] this Sp set has low efficiency (over -25%%) for this mean jumpsize\n");
					}
				}
				else {
					//now_meanjumpsize = int(round(1.0*(sumjumpsize - dS[i-1]) / (i-1)))
					//next_meanjumpsize = int(round(1.0*(sumjumpsize) / (i)))
					Ii.SetInt32(i-1);
					now_meanjumpsize = sumjumpsize; now_meanjumpsize.Sub(&dS[i-1]); now_meanjumpsize.Div(&Ii);
					Ii.SetInt32(i);
					next_meanjumpsize = sumjumpsize; next_meanjumpsize.Div(&Ii);

					//if  optimalmeanjumpsize - now_meanjumpsize <= next_meanjumpsize - optimalmeanjumpsize : 
					Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize);
					Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize);

					//len100perc = 60
					Int len100perc; len100perc.SetInt32(60);
					//size1perc = (next_meanjumpsize - now_meanjumpsize)//len100perc
					Int size1perc; size1perc.Sub(&next_meanjumpsize, &now_meanjumpsize);
					size1perc.Div(&len100perc);
					printf("[i] Sp[%d]|", i-1);
					//, '-'*(abs(optimalmeanjumpsize - now_meanjumpsize)//size1perc)
					Isub1.Abs(); Isub1.Div(&size1perc);
					for (uint32_t j = 0; j < Isub1.GetInt32() ; ++j) printf("-");
					printf("J");
					//, '-'*(abs(next_meanjumpsize - optimalmeanjumpsize)//size1perc)
					Isub2.Abs(); Isub2.Div(&size1perc);
					for (uint32_t j = 0; j < Isub2.GetInt32() ; ++j) printf("-");
					printf("|Sp[%d]\n", i);
					//if (1.0 * abs(next_meanjumpsize - optimalmeanjumpsize) / abs(optimalmeanjumpsize - now_meanjumpsize) >= 0.25) {
					Isub1.Sub(&next_meanjumpsize, &optimalmeanjumpsize); Isub1.Abs();
					Isub2.Sub(&optimalmeanjumpsize, &now_meanjumpsize); Isub2.Abs();
					if ((float) Isub1.GetInt32() / Isub2.GetInt32() >= 0.25) {
						printf("[i] this Sp set has low efficiency (over -25%%) for this mean jumpsize\n");
					}
				}
			}

			if (flag_verbose > 0)
				printf("[JmaxofSp] Sp[%d]=%s nearer to optimal mean jumpsize of Sp set\n", i
					, now_meanjumpsize.GetBase16().c_str()
				);
						
			return i;
		}
	}
	return 0;
}


// ----------------------------------------------------------------------------

void VanitySearch::Search(bool useGpu, std::vector<int> gpuId, std::vector<int> gridSize) {

  flag_endOfSearch = false;

  nbCPUThread = nbThread;
  nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
  totalRW = 0;
  
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  /////////////////////////////////////////////////
  // profiles load
  xU = 0;    
  xV = 0;
  // number kangaroos of herd T/W
  if (nbCPUThread == 1 || nbCPUThread == 2) {
	  xU = xV = 1;
	  xUV = 1;
	  bxU.SetInt32(xU); bxV.SetInt32(xV); bxUV.Mult(&bxU, &bxV);
  } 
  else if (nbCPUThread >= 4) {
	  // odd int
	  xU = (nbCPUThread/2)-1;
	  xV = (nbCPUThread/2)+1;
	  xUV = (uint64_t)xU * (uint64_t)xV;
	  bxU.SetInt32(xU); bxV.SetInt32(xV); bxUV.Mult(&bxU, &bxV);

	  //printf("[+] NO Recalc Sp-table of multiply UV\n");
	  if (1) {
		  for (int k = 0; k < 256; ++k) {
			  dS[k].Mult(&bxUV);
			  Sp[k] = secp->MultKAffine(bxUV, Sp[k]);
		  }
		  if (flag_verbose > 0)
			  printf("[+] recalc Sp-table of multiply UV\n");
	  }

  }

  printf("[UV] U*V=%d*%d=%llu (0x%02llx)\n", xU, xV, xUV, xUV);


  /////////////////////////////////////////////////
  //

  Int midJsize;
  midJsize.SetInt32(0);
  //printf("[i] 0x%064s\n", midJsize.GetBase16().c_str());

  // mean jumpsize
  if (xU==1 && xV==1) {
	  // by Pollard ".. The best choice of m (mean jump size) is w^(1/2)/2 .."
	  //midJsize = (Wsqrt//2)+1
	  //midJsize = int(round(1.0*Wsqrt / 2))
	  midJsize = bnWsqrt; midJsize.ShiftR(1);
  } 
  else {
	  // expected of 2w^(1/2)/cores jumps
	  //midJsize = int(round(1.0*((1.0*W/(xU*xV))**0.5)/2))
	  //midJsize = int(round(1.0*(xU+xV)*Wsqrt/4));
	  midJsize = bnWsqrt; midJsize.Mult((uint64_t)(xU+xV)); midJsize.ShiftR(2);
	  //midJsize = int(round(1.0*Wsqrt/2));

  }

  JmaxofSp = (uint64_t)getJmaxofSp(midJsize, dS);

  //sizeJmax = 2**JmaxofSp
  sizeJmax = dS[JmaxofSp];

  /////////////////////////////////////////////////
  // discriminator of distinguished points

  int pow2dp = (pow2W/2)-2;
  if (pow2dp > 24) {//if (pow2dp > 63) {
	  //printf("\n[FATAL_ERROR] overflow DPmodule! (uint64_t)\n");
	  //exit(EXIT_FAILURE);
	  printf("[i] Old DPmodule: 2^%d \n", pow2dp);
	  pow2dp = 24;//pow2dp = 63;
	  printf("[i] New DPmodule: 2^%d \n", pow2dp);
  }
  DPmodule = (uint64_t)1<<pow2dp;
  if (flag_verbose > 0)
	  printf("[DPmodule] 2^%d = %llu (0x%016llX) \n", pow2dp, DPmodule, DPmodule);


  /////////////////////////////////////////////////
  // create T/W herd

  //TH_PARAM *params = (TH_PARAM *)calloc((nbThread), sizeof(TH_PARAM));
  TH_PARAM *params = (TH_PARAM *)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

  // dK - int, sum distance traveled
  // Kp - point, sum distance traveled
  // add !!!
  Int KtmpR;
  KtmpR.Rand(pow2W - 1);

  // Tame herd, generate start points
  for (int k = 0; k < xU; ++k) {

	  params[k].type = false; // false - Tame, true - Wild
  
	  //dT.append(M + k*xV)  
	  params[k].dK = bnM;
	  Int Ktmp; Ktmp.SetInt32(k); Ktmp.Mult(&bxV);
	  params[k].dK.Add(&Ktmp);
	  
	  // add !!!
	  params[k].dK.Add(&KtmpR);
	  

	  if (flag_verbose > 1)	printf(" dT[%d] 0x%064s \n", k, params[k].dK.GetBase16().c_str());

	  //Tp.append(mul_ka(dT[k]))
	  //params[k].Kp = secp->MultKAffine(params[k].dK, secp->G);
	  params[k].Kp = secp->ComputePubKey(&params[k].dK);
	  
  }


  // Wild herd, generate start points
  for (int k = xU; k < (xU+xV) ; ++k) {

	  params[k].type = true; // false - Tame, true - Wild
	  //dW.append(1 + xU*k)
	  
	  params[k].dK.SetInt32(1);
	  Int Ktmp; Ktmp.SetInt32(k-xU); Ktmp.Mult(&bxU);
	  params[k].dK.Add(&Ktmp);
	  
	  // add !!!
	  //params[k].dK.Add(&KtmpR);
	  
	  
	  if (flag_verbose > 1)	printf(" dW[%d] 0x%064s \n", k, params[k].dK.GetBase16().c_str());

	  //Wp.append(add_a(W0p, mul_ka(dW[k])))
	  //params[k].Kp = secp->AddAffine(targetPubKey, secp->MultKAffine(params[k].dK, secp->G));
	  // ;)
	  Point pdK = secp->ComputePubKey(&params[k].dK);
	  params[k].Kp = secp->AddAffine(targetPubKey, pdK);
	  //params[k].Kp = secp->AddAffine(targetPubKey, secp->ComputePubKey(&params[k].dK));
	  	  
  }

  printf("[+] %dT+%dW kangaroos - ready\n", xU, xV);


  /////////////////////////////////////////////////
  // Launch threads

  if (1) {//if (useGpu) {

	  printf("[CPU] threads: %d \n", nbThread);

	  // Tame herd, start
	  for (int k = 0; k < xU; ++k) {

		  params[k].obj = this;
		  params[k].threadId = k;
		  params[k].isRunning = true;

		  #ifdef WIN64
		  DWORD thread_id;
		  CreateThread(NULL, 0, _FindKey, (void*)(params+k), 0, &thread_id);
		  ghMutex = CreateMutex(NULL, FALSE, NULL);
		  #else
		  pthread_t thread_id;
		  pthread_create(&thread_id, NULL, &_FindKey, (void*)(params+k));
		  ghMutex = PTHREAD_MUTEX_INITIALIZER;
		  #endif
	  }

	  // Wild herd, start
	  for (int k = xU; k < (xU+xV); ++k) {

		  params[k].obj = this;
		  params[k].threadId = k;
		  params[k].isRunning = true;

		  #ifdef WIN64
		  DWORD thread_id;
		  CreateThread(NULL, 0, _FindKey, (void*)(params+k), 0, &thread_id);
		  ghMutex = CreateMutex(NULL, FALSE, NULL);
		  #else
		  pthread_t thread_id;
		  pthread_create(&thread_id, NULL, &_FindKey, (void*)(params+k));
		  ghMutex = PTHREAD_MUTEX_INITIALIZER;
		  #endif
	  }

  } 
  if (useGpu) {//if (1) {
  
	printf("[GPU] threads: %d Hang on to your hats... ;-)\n", nbGPUThread);
	
	// Launch GPU threads
  for (int i = 0; i < nbGPUThread; i++) {
    params[nbCPUThread+i].obj = this;
    params[nbCPUThread+i].threadId = 0x80L+i;
	params[nbCPUThread+i].isRunning = true;
    params[nbCPUThread+i].gpuId = gpuId[i];
	int x = gridSize[2 * i];
	int y = gridSize[2 * i + 1];
	if(!GPUEngine::GetGridSize(gpuId[i],&x,&y)) {
		return;
	}
	else {
		params[nbCPUThread+i].gridSizeX = x;
		params[nbCPUThread+i].gridSizeY = y;
	}
    totalRW += GPU_GRP_SIZE * x*y;
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKeyGPU, (void*)(params+(nbCPUThread+i)), 0, &thread_id);
	ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params+(nbCPUThread+i)));
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
  }
  // Solver
  #ifdef WIN64
	DWORD thread_id;
	CreateThread(NULL, 0, _SolverGPU, (void*)(params+(nbCPUThread)), 0, &thread_id);
	ghMutex = CreateMutex(NULL, FALSE, NULL);
  #else
	pthread_t thread_id;
	pthread_create(&thread_id, NULL, &_SolverGPU, (void*)(params+(nbCPUThread)));
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
  #endif  
  //   
  }

  // Wait that all threads have started
  while (!hasStarted(params)) {
	Timer::SleepMillis(500);
  }

  /////////////////////////////////////////////////
  // indicator, progress, time, info...

  #ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
  #endif
  
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;

  //time vars
  double t0, t1, timestart;
  t0 = t1 = timestart = Timer::get_tick();
  time_t timenow, timepass;
  char timebuff[255 + 1] = { 0 };

  uint64_t countj_all=0, lastCount=0;
  memset(countj, 0, sizeof(countj));

  // Key rate smoothing filter
  #define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;
  
  double keyRate = 0.0;
  double gpuKeyRate = 0.0;
  
  memset(lastkeyRate, 0, sizeof(lastkeyRate));
  memset(lastGpukeyRate,0,sizeof(lastkeyRate));
  
  // Wait that all threads have started
  while (!hasStarted(params)) {
    Timer::SleepMillis(500);
  }
  
  int n_rot = 1;
  Int bnTmp, bnTmp2;

  // wait solv..
  while (isAlive(params)) {

	  int delay = 2000;
	  while (isAlive(params) && delay>0) {
		  Timer::SleepMillis(500);
		  delay -= 500;
	  }

	  t1 = Timer::get_tick();
	  //countj_all = getCountJ();
	  uint64_t gpuCount = getGPUCount();
	  countj_all = getCountJ() + gpuCount;
	  
	  keyRate = (double)(countj_all - lastCount) / (t1 - t0);
	  gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
	  lastkeyRate[filterPos%FILTER_SIZE] = keyRate;
	  lastGpukeyRate[filterPos%FILTER_SIZE] = gpuKeyRate;
	  filterPos++;

	  // KeyRate smoothing
	  double avgKeyRate = 0.0;
	  double avgGpuKeyRate = 0.0;
	  uint32_t nbSample;
	  for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
		  avgKeyRate += lastkeyRate[nbSample];
		  avgGpuKeyRate += lastGpukeyRate[nbSample];
	  }
	  avgKeyRate /= (double)(nbSample);
	  avgGpuKeyRate /= (double)(nbSample);

	  if (isAlive(params)) {
			
		  printf("\r");

		  if     (n_rot == 1) { printf("[\\]"); n_rot = 2; }
		  else if(n_rot == 2) { printf("[|]"); n_rot = 3; }
		  else if(n_rot == 3) { printf("[/]"); n_rot = 0; }
		  else                { printf("[-]"); n_rot = 1; }

		  timepass = (time_t)(t1-timestart);
		  //passtime(timebuff, sizeof(timebuff), gmtime(&timepass));
		  bnTmp.SetInt32((uint32_t)timepass);
		  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
		  printf("[%s ;", timebuff); // timelost

		  //printf(" %6s j/s;", prefSI_double(buff_s32, sizeof(buff_s32), (double)avgKeyRate)); // speed
		  printf(" %6s j/s; [GPU %.2f Mj/s]", prefSI_double(buff_s32, sizeof(buff_s32), (double)avgKeyRate), avgGpuKeyRate / 1000000.0); // speed
		  //bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)avgKeyRate);
		  //printf(" %6s j/s;", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp) ); // speed

		  //printf(" %6sj", prefSI_double(buff_s32, sizeof(buff_s32), (double)countj_all)); // count jumps
		  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all);
		  printf(" %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // count jumps

		  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Mult((uint64_t)100); 
		  //bnTmp.ShiftR(pow2Wsqrt+1);
		  //printf(" %3s.0%%;", bnTmp.GetBase10().c_str()); // percent progress, expected 2w^(1/2) jumps
		  bnTmp.ShiftR(pow2Wsqrt-2); double percprog = (double)bnTmp.GetInt32(); percprog /= 2*2*2;
		  printf(" %5.1f%%;", percprog); // percent progress, expected 2w^(1/2) jumps

		  printf(" dp/kgr=%.1lf;", (double)(countDT+countDW)/2 ); // distinguished points

		  bnTmp.SetInt32(1); bnTmp.ShiftL(1 + pow2Wsqrt); bnTmp.Sub((uint64_t)countj_all);
		  if (bnTmp.IsPositive()) {
			  bnTmp2.SetInt32(0); bnTmp2.SetQWord(0, (uint64_t)avgKeyRate);
			  bnTmp.Div(&bnTmp2);
			  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
			  printf("%s ]", timebuff); // timeleft
		  }
		  else {
			  bnTmp.SetInt32(0);
			  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
			  printf("%s ]", timebuff); // timeleft
		  }

		  printf("  ");
		  
		  /*
		  // PubKeys list End of search
		  int time_out = 180;// seconds
		  int max_perc = 300;// max percent
		  if (time_out > 0) {
			if ((int)timepass >= time_out && (int)percprog > max_perc) {
				printf("\n[i] End of search! Maximum percent: %d and time: %d seconds \n", max_perc, time_out);
				flag_endOfSearch = true;
				t0 = t1 = timestart = Timer::get_tick();
				timepass = time(NULL);
				countj_all = 0;
				lastCount = 0;
				gpuCount = 0;
				lastGPUCount = 0;
			} 
		  }
		  */
	  }
	  
	  if (rekey > 0) {
		if ((gpuCount - lastRekey) > (1000000 * rekey)) {
			// Rekey request
			rekeyRequest(params);
			lastRekey = gpuCount;
		}
	  }
	  
	  lastCount = countj_all;
	  lastGPUCount = gpuCount;
	  t0 = t1;
  }
  printf("\n");

  t1 = Timer::get_tick();
  timepass = (time_t)(t1 - timestart);
  if (timepass == 0) timepass = 1;

  /////////////////////////////////////////////////
  // print prvkey

  if (nbCPUThread > 0){
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  printf("[prvkey#%d] 0x%064s \n", pow2U, resultPrvKey.GetBase16().c_str());

  if (checkPrivKeyCPU(resultPrvKey, targetPubKey)) {
	  if (outputFile.length() > 0 
		  && output(resultPrvKey.GetBase16().c_str()
			  + string(":") + string("04") + targetPubKey.x.GetBase16().c_str() + targetPubKey.y.GetBase16().c_str()
			 )
		)  
		printf("[i] success saved pair prvkey:pubkey to file '%s'\n", outputFile.c_str());
  }
  else {
	  printf("[pubkey-check] failed!\n");
  }
  }

  /////////////////////////////////////////////////
  // final stat

  printf("[i]");

  t1 = Timer::get_tick();
  //countj_all = getCountJ();
  countj_all = getCountJ() + getGPUCount();

  ////timepass = (time_t)(t1 - timestart);
  printf(" %6s j/s;", prefSI_double(buff_s32, sizeof(buff_s32), (double)(countj_all / timepass))); // speed
  //bnTmp2.SetInt32((uint32_t)timepass); 
  //bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Div(&bnTmp2);
  //printf(" %6s j/s;", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // speed

  //printf(" %6sj", prefSI_double(buff_s32, sizeof(buff_s32), (double)countj_all)); // count jumps
  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all);
  printf(" %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // count jumps

  bnTmp.SetInt32(1); bnTmp.ShiftL(1 + pow2Wsqrt);
  printf(" of %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // expected 2w^(1/2) jumps

  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Mult((uint64_t)100);
  //bnTmp.ShiftR(pow2Wsqrt+1);
  //printf(" %3s.0%%;", bnTmp.GetBase10().c_str()); // percent progress, expected 2w^(1/2) jumps
  bnTmp.ShiftR(pow2Wsqrt - 2); double percprog = (double)bnTmp.GetInt32(); percprog /= 2 * 2 * 2;
  printf(" %5.1f%%;", percprog); // percent progress, expected 2w^(1/2) jumps

  printf(" DP %dT+%dW=%llu+%llu=%llu; dp/kgr=%.1lf;"
	  , xU, xV, countDT, countDW, countDT+countDW, (double)(countDT+countDW)/2
  ); // distinguished points

  printf("\n");

  /////////////////////////////////////////////////

  //timepass = (time_t)(t1 - timestart);
  bnTmp.SetInt32((uint32_t)timepass);
  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
  printf("[runtime] %s \n", timebuff);

  /////////////////////////////////////////////////

  free(params);
  free(DPht);

  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");
  timenow = time(NULL);
  strftime(timebuff, sizeof(timebuff), "%d %b %Y %H:%M:%S", gmtime(&timenow));
  printf("[DATE(utc)] %s\n", timebuff);
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");
  printf("[x] EXIT\n");exit(EXIT_SUCCESS);
}

// ----------------------------------------------------------------------------

void VanitySearch::ReWriteFiles(){
	
	FILE *f1 = fopen("tame-1.txt", "w");
	fclose(f1);
	FILE *f2 = fopen("wild-1.txt", "w");
	fclose(f2);
	FILE *f3 = fopen("tame-2.txt", "w");
	fclose(f3);
	FILE *f4 = fopen("wild-2.txt", "w");
	fclose(f4);

}

// ----------------------------------------------------------------------------

bool VanitySearch::SolverChmod(){
	
	string comm_chmod = "chmod 755 ./solver-1.py";
  const char* schmod = comm_chmod.c_str();
  bool chm = system(schmod);
  // 
  string comm_chmod2 = "chmod 755 ./solver-2.py";
  const char* schmod2 = comm_chmod2.c_str();
  bool chm2 = system(schmod2);
  // 
  string comm_chmod3 = "chmod 755 ./solver-3.py";
  const char* schmod3 = comm_chmod3.c_str();
  bool chm3 = system(schmod3);
  // 
  string comm_chmod4 = "chmod 755 ./solver-4.py";
  const char* schmod4 = comm_chmod4.c_str();
  bool chm4 = system(schmod4);
  
  if (chm && chm2 && chm3 && chm4) {
	return true;
  } else {
	return false;
  }

}

// ----------------------------------------------------------------------------
/*
string VanitySearch::GetHex(vector<unsigned char> &buffer) {

  string ret;

  char tmp[128];
  for (int i = 0; i < (int)buffer.size(); i++) {
    sprintf(tmp,"%02X",buffer[i]);
    ret.append(tmp);
  }

  return ret;

}
*/
