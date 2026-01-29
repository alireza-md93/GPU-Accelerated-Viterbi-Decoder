#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"

template<Metric metricType>
struct trellis;

//two packed 16-bit values
template<>
struct trellis<Metric::B16>{
	unsigned int pm;
	unsigned int pp;

	__device__ trellis(): pm(0), pp(0) {}
};

template<>
struct trellis<Metric::B32>{
	int pm0;
	int pm1;
	unsigned int pp0;
	unsigned int pp1;

	__device__ trellis(): pm0(0), pm1(0), pp0(0), pp1(0) {}
};

template<Metric metricType>
__device__ void forwardACS(int stage, trellis<metricType>& old, trellis<metricType>& now, decPack_t<metricType> pathPrev[][1<<(CL-1)], metric_t<metricType> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){}

template<>
__device__ void forwardACS<Metric::B16>(int stage, trellis<Metric::B16>& old, trellis<Metric::B16>& now, decPack_t<Metric::B16> pathPrev[][1<<(CL-1)], metric_t<Metric::B16> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int ind = stage % shmemWidth;
	int stageSE = stage % (CL-1);

	if(stageSE < CL-6){
		unsigned int bms = (uint16_t)branchMetric[ind][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[ind][allBmInd0&3];

		unsigned int pmRev = __funnelshift_l(old.pm, old.pm, 16);

		//------------------------- DPX ---------------------------
		// now.pm = __viaddmax_s16x2(old.pm, __vadd2(bms, bms), pmRev);
		// unsigned int cond = __vcmpeq2(now.pm, pmRev);
		// now.pm = __vsub2(now.pm, bms);
		// unsigned int permMap = (((cond>>8) & 0x00002222) ^ 0x00002200) | 0x00001010;
		// now.pp = __byte_perm(old.pp, 0, permMap);
		// cond &= 0x00010001;
		// cond ^= 1;
		// now.pp = (now.pp << 1) | cond;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool condH, condL;
		now.pm = __vibmax_s16x2(__vadd2(old.pm, bms), __vadd2(pmRev, __vneg2(bms)), &condH, &condL);
		unsigned int permMap = (condH?0x00002200U:0U) | (condL?0U:0x00000022U) | 0x00001010;
		now.pp = __byte_perm(old.pp, 0, permMap);
		unsigned int padd = (condH?0U:0x00010000U) | (condL?0x00000001U:0U);
		now.pp = (now.pp << 1) | padd; 
		//------------------------- Reg ---------------------------

		old = now;
	}
	else{
		int laneMask = (1<<(stageSE-CL+6));
		now.pm = __shfl_xor_sync(0xffffffff, now.pm, laneMask);
		now.pp = __shfl_xor_sync(0xffffffff, now.pp, laneMask);
		
		unsigned int bms = (uint16_t)branchMetric[ind][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[ind][allBmInd1&3];
		
		//------------------------- DPX ---------------------------
		// unsigned int pmMax = __viaddmax_s16x2(old.pm, __vadd2(bms, bms), now.pm);
		// unsigned int cond = __vcmpeq2(pmMax, now.pm);
		// now.pm = __vsub2(pmMax, bms);
		// unsigned int permMap = ((cond>>8) & 0x00004444) | 0x00003210;
		// now.pp = __byte_perm(old.pp, now.pp, permMap);
		// cond &= 0x00010001;
		// cond ^= (tx & laneMask) ? 0x00010001 : 0;
		// now.pp = (now.pp << 1) | cond;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool condH, condL;
		now.pm = __vibmax_s16x2(__vadd2(old.pm, bms), __vadd2(now.pm, __vneg2(bms)), &condH, &condL);
		unsigned int permMap = (condH?0x00004400U:0U) | (condL?0x00000044U:0U) | 0x00003210;
		now.pp = __byte_perm(now.pp, old.pp, permMap);
		unsigned int padd = (condH?0U:0x00010000U) | (condL?0U:0x00000001U);
		padd ^= (tx & laneMask) ? 0x00010001 : 0;
		now.pp = (now.pp << 1) | padd; 
		//------------------------- Reg ---------------------------

		old = now;
	}

	if(stage % 100 == 0){
		metric_t<Metric::B16> pmMin = static_cast<metric_t<Metric::B16>>(now.pm & 0x0000ffff);
		metric_t<Metric::B16> pmMax = static_cast<metric_t<Metric::B16>>(now.pm >> 16);
		if(pmMin > pmMax){
			metric_t<Metric::B16> temp = pmMin;
			pmMin = pmMax;
			pmMax = temp;
		}

		if(__any_sync(0xffffffff, pmMax > (1<<14))){
			for(unsigned int delta=16; delta>0; delta/=2)
				pmMin = min(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));

			unsigned int pmMinX2 = static_cast<unsigned int>(static_cast<uint16_t>(pmMin));
			pmMinX2 = (pmMinX2 << 16) | pmMinX2;
			pmMinX2 = __shfl_sync(0xffffffff, pmMinX2, 0);
			now.pm = __vsub2(now.pm, pmMinX2);
			old.pm = now.pm;

			// if(bx==0 && by==0 && ty==0 && tx==0){
			// 	printf("Stage %d: PM min=%d\n", stage, pmMin);
			// }
		}
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((ind+1) % 16 == 0){
		unsigned int padd = tx % (1<<stageSE);
		padd <<= (CL-1);
		unsigned int state0 = padd + tx;
		unsigned int state1 = state0 + (1U<<(CL-2));
		state0 >>= stageSE;
		state1 >>= stageSE;
		pathPrev[ind/16][state0] = now.pp >> 16;
		pathPrev[ind/16][state1] = now.pp  & 0x0000ffff;
		old.pp = 0;
		now.pp = 0;
	}
}

template<>
__device__ void forwardACS<Metric::B32>(int stage, trellis<Metric::B32>& old, trellis<Metric::B32>& now, decPack_t<Metric::B32> pathPrev[][1<<(CL-1)], metric_t<Metric::B32> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int ind = stage % shmemWidth;
	int stageSE = stage % (CL-1);

	if(stageSE < CL-6){
		int bm = branchMetric[ind][allBmInd0&3];

		//------------------------- DPX ---------------------------
		// now.pm0 = __viaddmax_s32(old.pm0, bm*2, old.pm1);
		// bool cond0 = (now.pm0 == old.pm1 ? 1 : 0);
		// now.pm0 = now.pm0 - bm;
		// now.pp0 = cond0 ? old.pp1 : old.pp0;
		// now.pp0 = (now.pp0 << 1) | cond0;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool cond0;
		now.pm0 = __vibmax_s32(old.pm1-bm, old.pm0+bm, &cond0);
		now.pp0 = cond0 ? old.pp1 : old.pp0;
		now.pp0 = (now.pp0 << 1) | cond0;
		//------------------------- Reg ---------------------------



		//------------------------- DPX ---------------------------
		// now.pm1 = __viaddmax_s32(old.pm1, bm*2, old.pm0);
		// bool cond1 = (now.pm1 == old.pm0 ? 0 : 1);
		// now.pm1 = now.pm1 - bm;
		// now.pp1 = cond1 ? old.pp1 : old.pp0;
		// now.pp1 = (now.pp1 << 1) | cond1;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool cond1;
		now.pm1 = __vibmax_s32(old.pm1+bm, old.pm0-bm, &cond1);
		now.pp1 = cond1 ? old.pp1 : old.pp0;
		now.pp1 = (now.pp1 << 1) | cond1;
		//------------------------- Reg ---------------------------

		old = now;
	}
	else{
		int laneMask = (1<<(stageSE-CL+6));
		now.pm0 = __shfl_xor_sync(0xffffffff, now.pm0, laneMask);
		now.pm1 = __shfl_xor_sync(0xffffffff, now.pm1, laneMask);
		now.pp0 = __shfl_xor_sync(0xffffffff, now.pp0, laneMask);
		now.pp1 = __shfl_xor_sync(0xffffffff, now.pp1, laneMask);
		
		int bm0 = branchMetric[ind][allBmInd0&3];
		int bm1 = branchMetric[ind][allBmInd1&3];
		
		//------------------------- DPX ---------------------------
		// int pmMax0 = __viaddmax_s32(old.pm0, bm0*2, now.pm0);
		// bool cond0 = (pmMax0 == now.pm0);
		// now.pm0 = pmMax0 - bm0;
		// now.pp0 = cond0 ? now.pp0 : old.pp0;
		// cond0 ^= ((tx & laneMask) != 0);
		// now.pp0 = (now.pp0 << 1) | cond0;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool cond0;
		now.pm0 = __vibmax_s32(now.pm0 - bm0, old.pm0 + bm0, &cond0);
		now.pp0 = cond0 ? now.pp0 : old.pp0;
		cond0 ^= ((tx & laneMask) != 0);
		now.pp0 = (now.pp0 << 1) | cond0;
		//------------------------- Reg ---------------------------



		//------------------------- DPX ---------------------------
		// int pmMax1 = __viaddmax_s32(old.pm1, bm1*2, now.pm1);
		// bool cond1 = (pmMax1 == now.pm1);
		// now.pm1 = pmMax1 - bm1;
		// now.pp1 = cond1 ? now.pp1 : old.pp1;
		// cond1 ^= ((tx & laneMask) != 0);
		// now.pp1 = (now.pp1 << 1) | cond1;
		//------------------------- DPX ---------------------------

		//------------------------- Reg ---------------------------
		bool cond1;
		now.pm1 = __vibmax_s32(now.pm1 - bm1, old.pm1 + bm1, &cond1);
		now.pp1 = cond1 ? now.pp1 : old.pp1;
		cond1 ^= ((tx & laneMask) != 0);
		now.pp1 = (now.pp1 << 1) | cond1;
		//------------------------- Reg ---------------------------

		old = now;
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((ind+1) % 32 == 0){
		unsigned int padd = tx % (1<<stageSE);
		padd <<= (CL-1);
		unsigned int state0 = padd + tx;
		unsigned int state1 = state0 + (1U<<(CL-2));
		state0 >>= stageSE;
		state1 >>= stageSE;
		pathPrev[ind/32][state0] = now.pp0;
		pathPrev[ind/32][state1] = now.pp1;
		old.pp0 = 0;
		old.pp1 = 0;
		now.pp0 = 0;
		now.pp1 = 0;
	}
}