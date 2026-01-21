#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"


//============================================= Trellis Structures =============================================
template<Metric metricType>
struct trellisPM{};

template<DecodeOut outputType>
struct trellisPP{};

//two packed 16-bit values
template<>
struct trellisPM<Metric::M_B16>{
	unsigned int val;

	__device__ trellisPM(): val(0) {}
};
template<>
struct trellisPP<DecodeOut::O_B16>{
	unsigned int val;

	__device__ trellisPP(): val(0) {}
};

template<>
struct trellisPM<Metric::M_B32>{
	int v0;
	int v1;

	__device__ trellisPM(): v0(0), v1(0) {}
};
template<>
struct trellisPP<DecodeOut::O_B32>{
	unsigned int v0;
	unsigned int v1;

	__device__ trellisPP(): v0(0), v1(0) {}
};

union half2_uint{
	__half2 fp;
	unsigned int ui;
};
template<>
struct trellisPM<Metric::M_FP16>{
	half2_uint val;

	__device__ trellisPM() {
		val.fp = __half2(__float2half(0.0f), __float2half(0.0f));
	}

	__device__ trellisPM<Metric::M_FP16>& operator=(const trellisPM<Metric::M_FP16>& other){
		this->val.ui = other.val.ui;
		return *this;
	}
};

//----------------------------------------------------------------

enum CondMode{MASK, BOOL};

template<CondMode condMode>
struct condACS{};

template<>
struct condACS<CondMode::MASK>
{
	unsigned int val;
};

template<>
struct condACS<CondMode::BOOL>
{
	bool v0;
	bool v1;
};


template<Metric metricType, CompMode compMode>
struct condModeEval{
	constexpr static CondMode value = 
		(metricType == Metric::M_B16 && compMode == CompMode::REG) ? CondMode::BOOL :
		(metricType == Metric::M_B16 && compMode == CompMode::DPX) ? CondMode::MASK :
		(metricType == Metric::M_B32) ? CondMode::BOOL :
		(metricType == Metric::M_FP16) ? CondMode::MASK :
		CondMode::BOOL; //default
};
template<Metric metricType, CompMode compMode>
struct condType{
	using type = condACS<condModeEval<metricType, compMode>::value>;
};

//============================================= PM update in self-paired mode =============================================

template<Metric metricType, CompMode compMode=CompMode::REG>
__device__ void selfPM(trellisPM<metricType>& old, trellisPM<metricType>& now, typename condType<metricType, compMode>::type& cond, metric_t<metricType> branchMetric[4], unsigned int& allBmInd0){}

template<>
__device__ void selfPM<Metric::M_B16, CompMode::DPX>(trellisPM<Metric::M_B16>& old, trellisPM<Metric::M_B16>& now, typename condType<Metric::M_B16, CompMode::DPX>::type& cond, metric_t<Metric::M_B16> branchMetric[4], unsigned int& allBmInd0){
	unsigned int bms = (uint16_t)branchMetric[allBmInd0&3];
	bms = (bms << 16) | bms;

	unsigned int pmRev = __funnelshift_l(old.val, old.val, 16);
	now.val = __viaddmax_s16x2(old.val, __vadd2(bms, bms), pmRev);
	cond.val = __vcmpeq2(now.val, pmRev) ^ 0xffff0000;
	now.val = __vsub2(now.val, bms);
}

template<>
__device__ void selfPM<Metric::M_B16, CompMode::REG>(trellisPM<Metric::M_B16>& old, trellisPM<Metric::M_B16>& now, typename condType<Metric::M_B16, CompMode::REG>::type& cond, metric_t<Metric::M_B16> branchMetric[4], unsigned int& allBmInd0){
	unsigned int bms = (uint16_t)branchMetric[allBmInd0&3];
	bms = (bms << 16) | bms;
	unsigned int pmRev = __funnelshift_l(old.val, old.val, 16);
	now.val = __vibmax_s16x2(__vadd2(pmRev, __vneg2(bms)), __vadd2(old.val, bms), &cond.v1, &cond.v0);
	cond.v1 = !(cond.v1);
}

//----------------------------------------------------------------

template<>
__device__ void selfPM<Metric::M_B32, CompMode::DPX>(trellisPM<Metric::M_B32>& old, trellisPM<Metric::M_B32>& now, typename condType<Metric::M_B32, CompMode::DPX>::type& cond, metric_t<Metric::M_B32> branchMetric[4], unsigned int& allBmInd0){
	int bm = branchMetric[allBmInd0&3];

	now.v0 = __viaddmax_s32(old.v0, bm*2, old.v1);
	cond.v0 = (now.v0 == old.v1 ? 1 : 0);
	now.v0 = now.v0 - bm;
	
	now.v1 = __viaddmax_s32(old.v1, bm*2, old.v0);
	cond.v1 = (now.v1 == old.v0 ? 0 : 1);
	now.v1 = now.v1 - bm;
}

template<>
__device__ void selfPM<Metric::M_B32, CompMode::REG>(trellisPM<Metric::M_B32>& old, trellisPM<Metric::M_B32>& now, typename condType<Metric::M_B32, CompMode::REG>::type& cond, metric_t<Metric::M_B32> branchMetric[4], unsigned int& allBmInd0){
	int bm = branchMetric[allBmInd0&3];

	now.v0 = __vibmax_s32(old.v1-bm, old.v0+bm, &cond.v0);
	now.v1 = __vibmax_s32(old.v1+bm, old.v0-bm, &cond.v1);
}

//----------------------------------------------------------------

template<>
__device__ void selfPM<Metric::M_FP16, CompMode::REG>(trellisPM<Metric::M_FP16>& old, trellisPM<Metric::M_FP16>& now, typename condType<Metric::M_FP16, CompMode::REG>::type& cond, metric_t<Metric::M_FP16> branchMetric[4], unsigned int& allBmInd0){
	__half2 bms = __half2(branchMetric[allBmInd0&3], branchMetric[allBmInd0&3]);

	half2_uint pmRev;
	pmRev.ui = __funnelshift_l(old.val.ui, old.val.ui, 16);

	__half2 pm1 = __hadd2(old.val.fp, bms);
	__half2 pm2 = __hsub2(pmRev.fp, bms);
	now.val.fp = __hmax2(pm1, pm2);
	cond.val = __hlt2_mask(pm1, pm2) ^ 0xffff0000;
}

//============================================= PP update in self-paired mode =============================================

template<DecodeOut outputType, CondMode condMode>
__device__ void selfPP(trellisPP<outputType>& old, trellisPP<outputType>& now, condACS<condMode>& cond){}

template<>
__device__ void selfPP<DecodeOut::O_B16, CondMode::MASK>(trellisPP<DecodeOut::O_B16>& old, trellisPP<DecodeOut::O_B16>& now, condACS<CondMode::MASK>& cond){
	unsigned int permMap = ((cond.val>>8) & 0x00002222) | 0x00001010;
	now.val = __byte_perm(old.val, 0, permMap);
	cond.val &= 0x00010001;
	now.val = (now.val << 1) | cond.val;
}

template<>
__device__ void selfPP<DecodeOut::O_B16, CondMode::BOOL>(trellisPP<DecodeOut::O_B16>& old, trellisPP<DecodeOut::O_B16>& now, condACS<CondMode::BOOL>& cond){
	unsigned int permMap = (cond.v0?0x00000022U:0U) | (cond.v1?0x00002200U:0U) | 0x00001010;
	now.val = __byte_perm(old.val, 0, permMap);
	unsigned int padd = (cond.v0?0x00000001U:0U) | (cond.v1?0x00010000U:0U);
	now.val = (now.val << 1) | padd; 
}

//----------------------------------------------------------------

template<>
__device__ void selfPP<DecodeOut::O_B32, CondMode::MASK>(trellisPP<DecodeOut::O_B32>& old, trellisPP<DecodeOut::O_B32>& now, condACS<CondMode::MASK>& cond){
	now.v0 = (cond.val & 0x00000001) ? old.v1 : old.v0;
	now.v0 = (now.v0 << 1) | cond.val & 0x00000001;
	
	now.v1 = (cond.val & 0x00010000) ? old.v1 : old.v0;
	now.v1 = (now.v1 << 1) | ((cond.val & 0x00010000)>>16);
}

template<>
__device__ void selfPP<DecodeOut::O_B32, CondMode::BOOL>(trellisPP<DecodeOut::O_B32>& old, trellisPP<DecodeOut::O_B32>& now, condACS<CondMode::BOOL>& cond){
	now.v0 = cond.v0 ? old.v1 : old.v0;
	now.v0 = (now.v0 << 1) | cond.v0;
	
	now.v1 = cond.v1 ? old.v1 : old.v0;
	now.v1 = (now.v1 << 1) | cond.v1;
}

//============================================= PM update in SE mode =============================================

template<Metric metricType, CompMode compMode=CompMode::REG>
__device__ void pairPM(trellisPM<metricType>& old, trellisPM<metricType>& now, typename condType<metricType, compMode>::type& cond, metric_t<metricType> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){}

template<>
__device__ void pairPM<Metric::M_B16, CompMode::DPX>(trellisPM<Metric::M_B16>& old, trellisPM<Metric::M_B16>& now, typename condType<Metric::M_B16, CompMode::DPX>::type& cond, metric_t<Metric::M_B16> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	unsigned int bms = (uint16_t)branchMetric[allBmInd1&3];
	bms = (bms << 16) | (uint16_t)branchMetric[allBmInd0&3];
	
	unsigned int pmMax = __viaddmax_s16x2(old.val, __vadd2(bms, bms), now.val);
	cond.val = __vcmpeq2(pmMax, now.val);
	now.val = __vsub2(pmMax, bms);
}
	
template<>
__device__ void pairPM<Metric::M_B16, CompMode::REG>(trellisPM<Metric::M_B16>& old, trellisPM<Metric::M_B16>& now, typename condType<Metric::M_B16, CompMode::REG>::type& cond, metric_t<Metric::M_B16> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	unsigned int bms = (uint16_t)branchMetric[allBmInd1&3];
	bms = (bms << 16) | (uint16_t)branchMetric[allBmInd0&3];
	now.val = __vibmax_s16x2(__vadd2(now.val, __vneg2(bms)), __vadd2(old.val, bms), &cond.v1, &cond.v0);
}

//----------------------------------------------------------------

template<>
__device__ void pairPM<Metric::M_B32, CompMode::DPX>(trellisPM<Metric::M_B32>& old, trellisPM<Metric::M_B32>& now, typename condType<Metric::M_B32, CompMode::DPX>::type& cond, metric_t<Metric::M_B32> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int bm0 = branchMetric[allBmInd0&3];
	int bm1 = branchMetric[allBmInd1&3];
	
	int pmMax0 = __viaddmax_s32(old.v0, bm0*2, now.v0);
	cond.v0 = (pmMax0 == now.v0);
	now.v0 = pmMax0 - bm0;
	
	int pmMax1 = __viaddmax_s32(old.v1, bm1*2, now.v1);
	cond.v1 = (pmMax1 == now.v1);
	now.v1 = pmMax1 - bm1;
}

template<>
__device__ void pairPM<Metric::M_B32, CompMode::REG>(trellisPM<Metric::M_B32>& old, trellisPM<Metric::M_B32>& now, typename condType<Metric::M_B32, CompMode::REG>::type& cond, metric_t<Metric::M_B32> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int bm0 = branchMetric[allBmInd0&3];
	int bm1 = branchMetric[allBmInd1&3];

	now.v0 = __vibmax_s32(now.v0 - bm0, old.v0 + bm0, &cond.v0);
	now.v1 = __vibmax_s32(now.v1 - bm1, old.v1 + bm1, &cond.v1);
}

//----------------------------------------------------------------

template<>
__device__ void pairPM<Metric::M_FP16, CompMode::REG>(trellisPM<Metric::M_FP16>& old, trellisPM<Metric::M_FP16>& now, typename condType<Metric::M_FP16, CompMode::REG>::type& cond, metric_t<Metric::M_FP16> branchMetric[4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	__half2 bms = __half2(branchMetric[allBmInd0&3], branchMetric[allBmInd1&3]);
	__half2 pm1 = __hadd2(old.val.fp, bms);
	__half2 pm2 = __hsub2(now.val.fp, bms);
	now.val.fp = __hmax2(pm1, pm2);
	cond.val = __hlt2_mask(pm1, pm2);
}

//============================================= PP update in SE mode =============================================

template<DecodeOut outputType, CondMode condMode>
__device__ void pairPP(trellisPP<outputType>& old, trellisPP<outputType>& now, condACS<condMode>& cond, int& laneMask){}

template<>
__device__ void pairPP<DecodeOut::O_B16, CondMode::MASK>(trellisPP<DecodeOut::O_B16>& old, trellisPP<DecodeOut::O_B16>& now, condACS<CondMode::MASK>& cond, int& laneMask){
	unsigned int permMap = ((cond.val>>8) & 0x00004444) | 0x00003210;
	now.val = __byte_perm(old.val, now.val, permMap);
	cond.val &= 0x00010001;
	cond.val ^= (tx & laneMask) ? 0x00010001 : 0;
	now.val = (now.val << 1) | cond.val;
}

template<>
__device__ void pairPP<DecodeOut::O_B16, CondMode::BOOL>(trellisPP<DecodeOut::O_B16>& old, trellisPP<DecodeOut::O_B16>& now, condACS<CondMode::BOOL>& cond, int& laneMask){
	unsigned int permMap = (cond.v0?0x00000044U:0U) | (cond.v1?0x00004400U:0U) | 0x00003210;
	now.val = __byte_perm(old.val, now.val, permMap);
	unsigned int padd = (cond.v0?0x00000001U:0U) | (cond.v1?0x00010000U:0U);
	padd ^= (tx & laneMask) ? 0x00010001 : 0;
	now.val = (now.val << 1) | padd;
}

//----------------------------------------------------------------

template<>
__device__ void pairPP<DecodeOut::O_B32, CondMode::MASK>(trellisPP<DecodeOut::O_B32>& old, trellisPP<DecodeOut::O_B32>& now, condACS<CondMode::MASK>& cond, int& laneMask){
	now.v0 = (cond.val & 0x00000001) ? now.v0 : old.v0;
	now.v1 = (cond.val & 0x00010000) ? now.v1 : old.v1;

	cond.val ^= ((tx & laneMask) ? 0xffffffff : 0);

	now.v0 = (now.v0 << 1) | (cond.val & 0x00000001);
	now.v1 = (now.v1 << 1) | ((cond.val & 0x00010000)>>16);
}

template<>
__device__ void pairPP<DecodeOut::O_B32, CondMode::BOOL>(trellisPP<DecodeOut::O_B32>& old, trellisPP<DecodeOut::O_B32>& now, condACS<CondMode::BOOL>& cond, int& laneMask){
	now.v0 = cond.v0 ? now.v0 : old.v0;
	cond.v0 ^= ((tx & laneMask) != 0);
	now.v0 = (now.v0 << 1) | cond.v0;
	
	now.v1 = cond.v1 ? now.v1 : old.v1;
	cond.v1 ^= ((tx & laneMask) != 0);
	now.v1 = (now.v1 << 1) | cond.v1;
}

//============================================= PM normalization =============================================

template<Metric metricType>
__device__ void pmNormalization(trellisPM<metricType>& old, trellisPM<metricType>& now){}

template<>
__device__ void pmNormalization<Metric::M_B16>(trellisPM<Metric::M_B16>& old, trellisPM<Metric::M_B16>& now){
	metric_t<Metric::M_B16> pmMin = static_cast<metric_t<Metric::M_B16>>(now.val & 0x0000ffff);
	metric_t<Metric::M_B16> pmMax = static_cast<metric_t<Metric::M_B16>>(now.val >> 16);
	if(pmMin > pmMax){
		metric_t<Metric::M_B16> temp = pmMin;
		pmMin = pmMax;
		pmMax = temp;
	}

	if(__any_sync(0xffffffff, pmMax > static_cast<metric_t<Metric::M_B16>>(16000))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = min(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));

		unsigned int pmMinX2 = static_cast<unsigned int>(static_cast<uint16_t>(pmMin));
		pmMinX2 = (pmMinX2 << 16) | pmMinX2;
		pmMinX2 = __shfl_sync(0xffffffff, pmMinX2, 0);
		now.val = __vsub2(now.val, pmMinX2);
		old.val = now.val;

		// if(bx==0 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%d\n", stage, pmMin);
		// }
	}
}

template<>
__device__ void pmNormalization<Metric::M_B32>(trellisPM<Metric::M_B32>& old, trellisPM<Metric::M_B32>& now){
	metric_t<Metric::M_B32> pmMin = now.v0 > now.v1 ? now.v1 : now.v0;
	metric_t<Metric::M_B32> pmMax = now.v0 > now.v1 ? now.v0 : now.v1;

	if(__any_sync(0xffffffff, pmMax > static_cast<metric_t<Metric::M_B32>>(1000000000))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = min(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));
		
		pmMin = __shfl_sync(0xffffffff, pmMin, 0);
		now.v0 -= pmMin;
		now.v1 -= pmMin;
		old.v0 = now.v0;
		old.v1 = now.v1;

		// if(bx==0 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%d\n", stage, pmMin);
		// }
	}
}

template<>
__device__ void pmNormalization<Metric::M_FP16>(trellisPM<Metric::M_FP16>& old, trellisPM<Metric::M_FP16>& now){
	__half pmLimitMax = __half(500);
	metric_t<Metric::M_FP16> pmMin = __hmin(now.val.fp.x, now.val.fp.y);
	metric_t<Metric::M_FP16> pmMax = __hmax(now.val.fp.x, now.val.fp.y);

	if(__any_sync(0xffffffff, __hlt(pmLimitMax, pmMax))){
		for(unsigned int delta=16; delta>0; delta/=2)
			pmMin = __hmin(pmMin, __shfl_down_sync(0xffffffff, pmMin, delta));

		__half2 pmMinX2 = __half2(pmMin, pmMin);
		pmMinX2 = __shfl_sync(0xffffffff, pmMinX2, 0);
		now.val.fp = __hsub2(now.val.fp, pmMinX2);
		old.val.fp = now.val.fp;

		// for(unsigned int delta=16; delta>0; delta/=2)
		// 	pmMax = __hmax(pmMax, __shfl_down_sync(0xffffffff, pmMax, delta));
		// if(bx==20 && by==0 && ty==0 && tx==0){
		// 	printf("Stage %d: PM min=%f pmMax=%f\n", stage, static_cast<float>(pmMin), static_cast<float>(pmMax));
		// }
	}
}

//============================================= survivor path update =============================================

__device__ void stageToState(int stageSE, int& state0, int& state1){
	int padd = tx % (1<<stageSE);
	padd <<= (CL-1);
	state0 = padd + tx;
	state1 = state0 + (1U<<(CL-2));
	state0 >>= stageSE;
	state1 >>= stageSE;
}

template<DecodeOut outputType>
__device__ void pathPrevUpdate(int stageSE, int ppInd, trellisPP<outputType>& old, trellisPP<outputType>& now, decPack_t<outputType> pathPrev[][1<<(CL-1)]){}

template<>
__device__ void pathPrevUpdate<DecodeOut::O_B16>(int stageSE, int ppInd, trellisPP<DecodeOut::O_B16>& old, trellisPP<DecodeOut::O_B16>& now, decPack_t<DecodeOut::O_B16> pathPrev[][1<<(CL-1)]){
	int state0, state1;
	stageToState(stageSE, state0, state1);
	pathPrev[ppInd/16][state0] = now.val & 0x0000ffff;
	pathPrev[ppInd/16][state1] = now.val >> 16;
	old.val = 0;
	now.val = 0;
}

template<>
__device__ void pathPrevUpdate<DecodeOut::O_B32>(int stageSE, int ppInd, trellisPP<DecodeOut::O_B32>& old, trellisPP<DecodeOut::O_B32>& now, decPack_t<DecodeOut::O_B32> pathPrev[][1<<(CL-1)]){
	int state0, state1;
	stageToState(stageSE, state0, state1);
	pathPrev[ppInd/32][state0] = now.v0;
	pathPrev[ppInd/32][state1] = now.v1;
	old.v0 = 0;
	old.v1 = 0;
	now.v0 = 0;
	now.v1 = 0;
}

//============================================= warp-level shuffling =============================================

template<Metric metricType, DecodeOut outputType>
__device__ void contextExchange(int laneMask, trellisPM<metricType>& nowPM, trellisPP<outputType>& nowPP){}

template<>
__device__ void contextExchange<Metric::M_B16, DecodeOut::O_B16>(int laneMask, trellisPM<Metric::M_B16>& nowPM, trellisPP<DecodeOut::O_B16>& nowPP){
	nowPM.val = __shfl_xor_sync(0xffffffff, nowPM.val, laneMask);
	nowPP.val = __shfl_xor_sync(0xffffffff, nowPP.val, laneMask);
}
template<>
__device__ void contextExchange<Metric::M_B32, DecodeOut::O_B32>(int laneMask, trellisPM<Metric::M_B32>& nowPM, trellisPP<DecodeOut::O_B32>& nowPP){
	nowPM.v0 = __shfl_xor_sync(0xffffffff, nowPM.v0, laneMask);
	nowPM.v1 = __shfl_xor_sync(0xffffffff, nowPM.v1, laneMask);
	nowPP.v0 = __shfl_xor_sync(0xffffffff, nowPP.v0, laneMask);
	nowPP.v1 = __shfl_xor_sync(0xffffffff, nowPP.v1, laneMask);
}
template<>
__device__ void contextExchange<Metric::M_FP16, DecodeOut::O_B16>(int laneMask, trellisPM<Metric::M_FP16>& nowPM, trellisPP<DecodeOut::O_B16>& nowPP){
	nowPM.val.ui = __shfl_xor_sync(0xffffffff, nowPM.val.ui, laneMask);
	nowPP.val = __shfl_xor_sync(0xffffffff, nowPP.val, laneMask);
}

//============================================= General ACS function =============================================

template<Metric metricType, DecodeOut outputType, CompMode compMode=CompMode::REG>
__device__ void forwardACS(int stage, 
	trellisPM<metricType>& oldPM, trellisPP<outputType>& oldPP,
	trellisPM<metricType>& nowPM, trellisPP<outputType>& nowPP,
	decPack_t<outputType> pathPrev[][1<<(CL-1)], metric_t<metricType> branchMetric[][4], 
	unsigned int& allBmInd0, unsigned int& allBmInd1, int pmNormStride)
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
		contextExchange<metricType, outputType>(laneMask, nowPM, nowPP);

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
	// 	stageToState(stageSE, state0, state1);
	// 	if constexpr (metricType == Metric::M_B16 && outputType == DecodeOut::O_B16){
	// 		printf("=== stage:%d state:%d pm:%d pp:%x\n", stage, state0, nowPM.val&0xffff, nowPP.val&1);
	// 		printf("=== stage:%d state:%d pm:%d pp:%x\n", stage, state1, nowPM.val>>16, (nowPP.val>>16)&1);
	// 	}
	// 	else if constexpr (metricType == Metric::M_FP16 && outputType == DecodeOut::O_B16){
	// 		printf("=== stage:%d state:%d pm:%f pp:%x\n", stage, state0, __half2float(nowPM.val.fp.x), nowPP.val&1);
	// 		printf("=== stage:%d state:%d pm:%f pp:%x\n", stage, state1, __half2float(nowPM.val.fp.y), (nowPP.val>>16)&1);
	// 	}
	// 	else if constexpr (metricType == Metric::M_B32 && outputType == DecodeOut::O_B32){
	// 		printf("=== stage:%d state:%d pm:%d pp:%x\n", stage, state0, nowPM.v0, nowPP.v0&1);
	// 		printf("=== stage:%d state:%d pm:%d pp:%x\n", stage, state1, nowPM.v1, nowPP.v1&1);
	// 	}
	// }

	if((ppInd+1) % bpp<outputType> == 0) pathPrevUpdate<outputType>(stageSE, ppInd, oldPP, nowPP, pathPrev);
}