#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"
#include "viterbiACS_types.cuh"
#include "viterbiACS_SEen.cuh"
#include "viterbiACS_SEdis.cuh"

//============================================= General ACS function =============================================

template<Metric metricType, DecodeOut outputType, CompMode compMode, StateExchng stateEx>
struct ForwardACS{
};

template<Metric metricType, DecodeOut outputType, CompMode compMode>
struct ForwardACS<metricType, outputType, compMode, StateExchng::SE_EN>{
	trellisPM<metricType> oldPM;
	trellisPP<outputType> oldPP;
	trellisPM<metricType> nowPM;
	trellisPP<outputType> nowPP;
	metric_t<metricType> (*branchMetric)[4];
	unsigned int allBmInd0;
	unsigned int allBmInd1;
	int pmNormStride;
	__device__ void comp(int stage, decPack_t<outputType> (*pathPrev)[1<<(CL-1)])
	{
		int bmInd = stage % bmMemWidth;
		int ppInd = stage % forwardLen;
		int stageSE = stage % (CL-1);

		if(stageSE < CL-6){
			typename condType<metricType, compMode>::type cond;
			selfPM<metricType, compMode>(oldPM, nowPM, cond, branchMetric[bmInd], allBmInd0);
			selfPP<outputType, condModeEval<metricType, compMode>::value>(oldPP, nowPP, cond);
			oldPM = nowPM;
			oldPP = nowPP;
		}
		else{
			int laneMask = (1<<(stageSE-CL+6));
			pmExchange<metricType>(laneMask, nowPM);
			ppExchange<outputType>(laneMask, nowPP);

			typename condType<metricType, compMode>::type cond;
			pairPM<metricType, compMode>(oldPM, nowPM, cond, branchMetric[bmInd], allBmInd0, allBmInd1);
			pairPP<outputType, condModeEval<metricType, compMode>::value>(oldPP, nowPP, cond, laneMask);
			oldPM = nowPM;
			oldPP = nowPP;
		}

		if(stage % pmNormStride == 0) pmNormalization<metricType>(oldPM, nowPM);

		allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
		allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

		// if(bx==0 && ty==0){
		// 	int state0, state1;
		// 	int pm0, pm1;
		// 	int pp0, pp1;
			
		// 	stageToState(stageSE, state0, state1);

		// 	if constexpr (metricType == Metric::M_B16){
		// 		pm0 = nowPM.val & 0xffff;
		// 		pm1 = nowPM.val >> 16;
		// 	} else if constexpr (metricType == Metric::M_B32){
		// 		pm0 = nowPM.v0;
		// 		pm1 = nowPM.v1;
		// 	} else if constexpr (metricType == Metric::M_FP16){
		// 		pm0 = static_cast<int>(__half2float(nowPM.val.fp.x));
		// 		pm1 = static_cast<int>(__half2float(nowPM.val.fp.y));
		// 	}

		// 	if constexpr (outputType == DecodeOut::O_B16){
		// 		pp0 = nowPP.val & 1;
		// 		pp1 = (nowPP.val >> 16) & 1;
		// 	} else if constexpr (outputType == DecodeOut::O_B32){
		// 		pp0 = nowPP.v0 & 1;
		// 		pp1 = nowPP.v1 & 1;
		// 	}

		// 	printf("=== stage:%d state:%d pm:%d pp:%d\n", stage, state0, pm0, pp0);
		// 	printf("=== stage:%d state:%d pm:%d pp:%d\n", stage, state1, pm1, pp1);
		// }

		if((ppInd+1) % bpp<outputType> == 0) pathPrevUpdate<outputType>(stageSE, ppInd, oldPP, nowPP, pathPrev);
	}
};

template<Metric metricType, DecodeOut outputType, CompMode compMode>
struct ForwardACS<metricType, outputType, compMode, StateExchng::SE_DIS>{
	metric_t<metricType>* PM;
	decPack_t<outputType>* PP;
	metric_t<metricType> (*branchMetric)[4];
	unsigned int bmOffset;
	int pmNormStride;
	__device__ void comp(int stage, decPack_t<outputType> (*pathPrev)[1<<(CL-1)])
	{
		int bmInd = stage % bmMemWidth;
		int ppInd = stage % forwardLen;

		condACS<CondMode::BOOL> cond;
		metric_t<metricType> pm0, pm1;
		pmCalc_SEdis<metricType, compMode>(PM, pm0, pm1, cond, branchMetric[bmInd][bmOffset]);
		decPack_t<outputType> pp0 = cond.v0 ? PP[2*tx+1] : PP[2*tx];
		pp0 = (pp0 << 1) | cond.v0;
		decPack_t<outputType> pp1 = cond.v1 ? PP[2*tx+1] : PP[2*tx];
		pp1 = (pp1 << 1) | cond.v1;

		__syncwarp();
		
		PM[tx] = pm0;
		PM[tx+32] = pm1;
		PP[tx] = pp0;
		PP[tx+32] = pp1;
		
		__syncwarp();

		if(stage % pmNormStride == 0) pmNormalization_SEdis<metricType>(PM);

		// if(bx==0 && ty==0){
		// 	int state0, state1;
			
		// 	state0 = tx;
		// 	state1 = state0 + (1U<<(CL-2));
			
		// 	printf("=== stage:%d state:%d pm:%d pp:%d\n", stage, state0, pm0, pp0&1);
		// 	printf("=== stage:%d state:%d pm:%d pp:%d\n", stage, state1, pm1, pp1&1);
		// }

		if((ppInd+1) % bpp<outputType> == 0){
			pathPrev[ppInd/bpp<outputType>][tx] = pp0;
			pathPrev[ppInd/bpp<outputType>][tx+32] = pp1;
		}
	}
};