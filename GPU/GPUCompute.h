/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *da
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

// CUDA Kernel main function
// Compute SecpK1 keys
// We use affine coordinates for elliptic curve point (ie Z=1)

// Jump distance
__device__ __constant__ uint64_t dS[NB_JUMP][4];
// jump points
__device__ __constant__ uint64_t Spx[NB_JUMP][4];
__device__ __constant__ uint64_t Spy[NB_JUMP][4];

// -----------------------------------------------------------------------------------------

__device__ __noinline__ void SendPoint(uint64_t *pointx, uint64_t *distance, uint32_t type, uint32_t maxFound, uint32_t *out) {
	
	uint32_t pos; 
	uint32_t tid = (blockIdx.x*blockDim.x) + threadIdx.x; 
	//uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE); 
	
	// add Item
	pos = atomicAdd(out, 1);
	if (pos < maxFound) {
		// ITEM_SIZE 72 ITEM_SIZE32 18 or (ITEM_SIZE/4)
		out[pos*ITEM_SIZE32 + 1] = ((uint32_t *)pointx)[0];
		out[pos*ITEM_SIZE32 + 2] = ((uint32_t *)pointx)[1];
		out[pos*ITEM_SIZE32 + 3] = ((uint32_t *)pointx)[2];
		out[pos*ITEM_SIZE32 + 4] = ((uint32_t *)pointx)[3];
		out[pos*ITEM_SIZE32 + 5] = ((uint32_t *)pointx)[4];
		out[pos*ITEM_SIZE32 + 6] = ((uint32_t *)pointx)[5];
		out[pos*ITEM_SIZE32 + 7] = ((uint32_t *)pointx)[6];
		out[pos*ITEM_SIZE32 + 8] = ((uint32_t *)pointx)[7];
		out[pos*ITEM_SIZE32 + 9] = ((uint32_t *)distance)[0];
		out[pos*ITEM_SIZE32 + 10] = ((uint32_t *)distance)[1];
		out[pos*ITEM_SIZE32 + 11] = ((uint32_t *)distance)[2];
		out[pos*ITEM_SIZE32 + 12] = ((uint32_t *)distance)[3];
		out[pos*ITEM_SIZE32 + 13] = ((uint32_t *)distance)[4];
		out[pos*ITEM_SIZE32 + 14] = ((uint32_t *)distance)[5];
		out[pos*ITEM_SIZE32 + 15] = ((uint32_t *)distance)[6];
		out[pos*ITEM_SIZE32 + 16] = ((uint32_t *)distance)[7];
		out[pos*ITEM_SIZE32 + 17] = ((uint32_t)type);
		out[pos*ITEM_SIZE32 + 18] = ((uint32_t)tid);		
	}	
}

#define SetPoint(pointx, distance, type) SendPoint(pointx, distance, type, maxFound, out);

// -----------------------------------------------------------------------------------------

__device__ void CheckDP(uint64_t *check_px, uint64_t *check_wpx, uint64_t *check_tk, uint64_t *check_wk, uint32_t type, uint64_t DPmodule, uint32_t maxFound, uint32_t *out) {
	
	// For Check
	//uint64_t outPx[4] = {0x4444444444444444ULL, 0x3333333333333333ULL, 0x2222222222222222ULL, 0x1111111111111111ULL};
	//uint64_t outKey[4] = {0x8888888888888888ULL, 0x7777777777777777ULL, 0x6666666666666666ULL, 0x5555555555555555ULL};
	//type = 2;// Wild
	//type = 1;// Tame
	
	uint64_t tame_px[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wild_px[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	Load256(tame_px, check_px);
	Load256(wild_px, check_wpx);
	
	// check, is it distinguished point ?
	if (type == 1) {// Tame 1
		if (tame_px[0] % DPmodule == 0) {
			//SetPoint(outPx, outKey, type);
			SetPoint(check_px, check_tk, type);
		}
	}
	
	if (type == 2) {// Wild 2
		if (wild_px[0] % DPmodule == 0) {
			//SetPoint(outPx, outKey, type);
			SetPoint(check_wpx, check_wk, type);
		}
	}
	
}

#define CHECK_POINT(check_px, check_wpx, check_tk, check_wk, type) CheckDP(check_px, check_wpx, check_tk, check_wk, type, DPmod, maxFound, out)

// -----------------------------------------------------------------------------------------

__device__ void ComputeKeys(uint64_t *Tkangaroos, uint64_t *Wkangaroos, uint64_t DPmod, uint32_t hop_modulo, uint32_t maxFound, uint32_t *out) {

	// The new code from 03.05.2020
	uint64_t dx[GPU_GRP_SIZE][4];
	uint64_t wdx[GPU_GRP_SIZE][4];
	
	uint64_t px[GPU_GRP_SIZE][4];// Tame points
	uint64_t py[GPU_GRP_SIZE][4];
	uint64_t wpx[GPU_GRP_SIZE][4];// Wild points
	uint64_t wpy[GPU_GRP_SIZE][4];
	
	uint64_t tk[GPU_GRP_SIZE][4];// Tame key
	uint64_t wk[GPU_GRP_SIZE][4];// Wild key
	
	uint32_t type = 0;
	uint32_t pw1 = 0;
	uint32_t pw2 = 0;
	
	uint64_t dy[4] = { 0ULL,0ULL,0ULL,0ULL };// Tame
	uint64_t _s[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t _p2[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t wdy[4] = { 0ULL,0ULL,0ULL,0ULL };// Wild
	uint64_t w_s[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t w_p2[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t rx[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t ry[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wrx[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t wry[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	// Load starting key
	__syncthreads();	
	LoadKangaroos(Tkangaroos, px, py, tk);
	LoadKangaroos(Wkangaroos, wpx, wpy, wk);
	
	// Sp-table Spx[256], Spy[256]
	// Distance dS[256]
	// uint32_t spin = 64;
	uint32_t j;
	for (j = 0; j < NB_SPIN; j++) {
	
		__syncthreads();
		
		// Get Tame dx
		for(uint32_t g = 0; g < GPU_GRP_SIZE; g++) {
			uint32_t pw1 = (px[g][0] % hop_modulo);//uint32_t pw1 = (px[g][0] % NB_JUMP);
			ModSub256(dx[g], px[g], Spx[pw1]);
		}
		
		_ModInvGrouped(dx);
		
		// Get Wild dx
		for(uint32_t g = 0; g < GPU_GRP_SIZE; g++) {
			uint32_t pw2 = (wpx[g][0] % hop_modulo);//uint32_t pw2 = (wpx[g][0] % NB_JUMP);
			ModSub256(wdx[g], wpx[g], Spx[pw2]);
		}
		
		_ModInvGrouped(wdx);
		
		for(uint32_t g = 0; g < GPU_GRP_SIZE; g++) { 
			
			// Get jump size Tame
			pw1 = (px[g][0] % hop_modulo);//pw1 = (px[g][0] % NB_JUMP);
			// Get jump size Wild
			pw2 = (wpx[g][0] % hop_modulo);//pw2 = (wpx[g][0] % NB_JUMP);
			
			// Add Hops Distance Tame
			//ModAdd256(tk[g], dS[pw1]);
			ModAdd256Order(tk[g], dS[pw1]);
			
			// Add Hops Distance Wild
			//ModAdd256(wk[g], dS[pw2]);
			ModAdd256Order(wk[g], dS[pw2]);
			
			// Tame Affine points addition
			
			ModSub256(dy, py[g], Spy[pw1]);
			_ModMult(_s, dy, dx[g]);
			_ModSqr(_p2, _s);
			
			ModSub256(rx, _p2, Spx[pw1]);
			ModSub256(rx, px[g]);
			
			ModSub256(ry, px[g], rx);
			_ModMult(ry, _s);
			ModSub256(ry, py[g]);
			
			Load256(px[g], rx);
			Load256(py[g], ry);
			
			type = 1;// Tame 1
			
			CHECK_POINT(px[g], wpx[g], tk[g], wk[g], type);
			
			// Wild Affine points addition
			
			ModSub256(wdy, wpy[g], Spy[pw2]);
			_ModMult(w_s, wdy, wdx[g]);
			_ModSqr(w_p2, w_s);
			
			ModSub256(wrx, w_p2, Spx[pw2]);
			ModSub256(wrx, wpx[g]);
			
			ModSub256(wry, wpx[g], wrx);
			_ModMult(wry, w_s);
			ModSub256(wry, wpy[g]);
			
			Load256(wpx[g], wrx);
			Load256(wpy[g], wry);
			
			type = 2;// Wild 2
			
			CHECK_POINT(px[g], wpx[g], tk[g], wk[g], type);
			
		}
		
	}
	
	// Update starting point
	__syncthreads();
	StoreKangaroos(Tkangaroos, px, py, tk);
	StoreKangaroos(Wkangaroos, wpx, wpy, wk);

}
// Hello ;) 
