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