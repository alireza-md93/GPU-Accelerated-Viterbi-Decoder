#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"
#include "viterbiACS_types.cuh"

//============================================= PM update in self-paired mode =============================================

template<Metric metricType, CompMode compMode=CompMode::REG>
__device__ void pmCalc_SEdis(metric_t<metricType>*& PM, metric_t<metricType>& pm0, metric_t<metricType>& pm1, condACS<CondMode::BOOL>& cond, metric_t<metricType> bm){}

// template<>
// __device__ void pmCalc_SEdis<Metric::M_B16, CompMode::DPX>(metric_t<Metric::M_B16>*& PM, metric_t<Metric::M_B16>& pm0, metric_t<Metric::M_B16>& pm1, condACS<CondMode::BOOL>& cond, metric_t<Metric::M_B16> bm){
// 	unsigned int bms = (uint16_t)bm;
// 	bms = (bms << 16) | bms;

// 	unsigned int* pms = reinterpret_cast<unsigned int*>(&PM[2*tx]);
// 	unsigned int pmRev = __funnelshift_l(pms, pms, 16);
// 	unsigned int pmNow = __viaddmax_s16x2(*pms, __vadd2(bms, bms), pmRev);
// 	cond.val = __vcmpeq2(pmNow, pmRev) ^ 0xffff0000;
// 	pmNow = __vsub2(pmNow, bms);
// 	pm0 = pmNow & 0xffff;
// 	pm1 = pmNow >> 16;
// }

template<>
__device__ void pmCalc_SEdis<Metric::M_B16, CompMode::REG>(metric_t<Metric::M_B16>*& PM, metric_t<Metric::M_B16>& pm0, metric_t<Metric::M_B16>& pm1, condACS<CondMode::BOOL>& cond, metric_t<Metric::M_B16> bm){
	pm0 = max(PM[2*tx+1]-bm, PM[2*tx]+bm);
	pm1 = max(PM[2*tx+1]+bm, PM[2*tx]-bm);
    cond.v0 = (PM[2*tx+1]-bm > PM[2*tx]+bm) ? 1 : 0;
    cond.v1 = (PM[2*tx+1]+bm > PM[2*tx]-bm) ? 1 : 0;
}

//----------------------------------------------------------------

template<>
__device__ void pmCalc_SEdis<Metric::M_B32, CompMode::DPX>(metric_t<Metric::M_B32>*& PM, metric_t<Metric::M_B32>& pm0, metric_t<Metric::M_B32>& pm1, condACS<CondMode::BOOL>& cond, metric_t<Metric::M_B32> bm){
	pm0 = __viaddmax_s32(PM[2*tx], bm*2, PM[2*tx+1]);
	cond.v0 = (pm0 == PM[2*tx+1] ? 1 : 0);
	pm0 = pm0 - bm;
	
	pm1 = __viaddmax_s32(PM[2*tx+1], bm*2, PM[2*tx]);
	cond.v1 = (pm1 == PM[2*tx] ? 0 : 1);
	pm1 = pm1 - bm;
}

template<>
__device__ void pmCalc_SEdis<Metric::M_B32, CompMode::REG>(metric_t<Metric::M_B32>*& PM, metric_t<Metric::M_B32>& pm0, metric_t<Metric::M_B32>& pm1, condACS<CondMode::BOOL>& cond, metric_t<Metric::M_B32> bm){
	pm0 = __vibmax_s32(PM[2*tx+1]-bm, PM[2*tx]+bm, &cond.v0);
	pm1 = __vibmax_s32(PM[2*tx+1]+bm, PM[2*tx]-bm, &cond.v1);
}

//----------------------------------------------------------------

template<>
__device__ void pmCalc_SEdis<Metric::M_FP16, CompMode::REG>(metric_t<Metric::M_FP16>*& PM, metric_t<Metric::M_FP16>& pm0, metric_t<Metric::M_FP16>& pm1, condACS<CondMode::BOOL>& cond, metric_t<Metric::M_FP16> bm){
	pm0 = __hmax(PM[2*tx+1]-bm, PM[2*tx]+bm);
	pm1 = __hmax(PM[2*tx+1]+bm, PM[2*tx]-bm);
	cond.v0 = (PM[2*tx+1]-bm > PM[2*tx]+bm) ? 1 : 0;
    cond.v1 = (PM[2*tx+1]+bm > PM[2*tx]-bm) ? 1 : 0;
}

//============================================= PM normalization =============================================

template<Metric metricType>
__device__ void pmNormalization_SEdis(metric_t<metricType>*& PM){}

template<>
__device__ void pmNormalization_SEdis<Metric::M_B16>(metric_t<Metric::M_B16>*& PM){

	metric_t<Metric::M_B16> pmMin = PM[tx] > PM[tx+32] ? PM[tx+32] : PM[tx];
	metric_t<Metric::M_B16> pmMax = PM[tx] > PM[tx+32] ? PM[tx] : PM[tx+32];

	if(__any_sync(0xffffffff, pmMax > static_cast<metric_t<Metric::M_B16>>(16000))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = min(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));

		pmMin = __shfl_sync(0xffffffff, pmMin, 0);
		PM[tx] -= pmMin;
		PM[tx+32] -= pmMin;

		// if(bx==0 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%d\n", stage, pmMin);
		// }
	}
}

template<>
__device__ void pmNormalization_SEdis<Metric::M_B32>(metric_t<Metric::M_B32>*& PM){
	metric_t<Metric::M_B32> pmMin = PM[tx] > PM[tx+32] ? PM[tx+32] : PM[tx];
	metric_t<Metric::M_B32> pmMax = PM[tx] > PM[tx+32] ? PM[tx] : PM[tx+32];

	if(__any_sync(0xffffffff, pmMax > static_cast<metric_t<Metric::M_B32>>(1000000000))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = min(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));
		
		pmMin = __shfl_sync(0xffffffff, pmMin, 0);
		PM[tx] -= pmMin;
		PM[tx+32] -= pmMin;

		// if(bx==0 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%d\n", stage, pmMin);
		// }
	}
}

template<>
__device__ void pmNormalization_SEdis<Metric::M_FP16>(metric_t<Metric::M_FP16>*& PM){
	__half pmLimitMax = __half(500);
	metric_t<Metric::M_FP16> pmMin = __hmin(PM[tx], PM[tx+32]);
	metric_t<Metric::M_FP16> pmMax = __hmax(PM[tx], PM[tx+32]);

	if(__any_sync(0xffffffff, __hlt(pmLimitMax, pmMax))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = __hmin(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));

		pmMin = __shfl_sync(0xffffffff, pmMin, 0);
		PM[tx] -= pmMin;
		PM[tx+32] -= pmMin;

		// for(unsigned int delta=16; delta>0; delta/=2)
		// 	pmMax = __hmax(pmMax, __shfl_down_sync(0xffffffff, pmMax, delta));
		// if(bx==20 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%f pmMax=%f\n", stage, static_cast<float>(pmMin), static_cast<float>(pmMax));
		// }
	}
}
