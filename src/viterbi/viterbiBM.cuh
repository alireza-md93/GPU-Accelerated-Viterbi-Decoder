#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"
#include <cstdint>

template<ChannelIn inputType>
struct bmCalcHelper;

template<ChannelIn inputType>
__device__ int bmCalcInt(int i, encPack_t<inputType>* coded, bmCalcHelper<inputType>& helper){}

//============================= HARD ============================

template<>
struct bmCalcHelper<ChannelIn::HARD>{
	int dataInd;
	int bmInd;
	uint32_t mask0;
	uint32_t mask1;
	uint32_t out0;
	uint32_t out1;
	__device__ bmCalcHelper(){
		dataInd = tx >> 2;
		bmInd = tx & 3;
		mask0 = 3U << (30 - 2*dataInd);
		mask1 = mask0 >> 16;
		out0 = static_cast<uint32_t>(bmInd) << (30 - 2*dataInd);
		out1 = out0 >> 16;
	}
};

template<>
__device__ int bmCalcInt<ChannelIn::HARD>(int i, encPack_t<ChannelIn::HARD>* coded, bmCalcHelper<ChannelIn::HARD>& helper){
	uint32_t mask, out;
	uint32_t data = coded[i>>4];
	mask = i%16 < 8 ? helper.mask0 : helper.mask1;
	out = i%16 < 8 ? helper.out0 : helper.out1;
	return (1 - __popc((data&mask) ^ out));
}


//============================= SOFT4 ============================

template<>
struct bmCalcHelper<ChannelIn::SOFT4>{
	int coeff;
	int dataInd;
	int bmInd;
	int shift;
	__device__ bmCalcHelper(){
		dataInd = tx >> 2;
		bmInd = tx & 3;
		shift = ((dataInd & 3) << 3);
		switch(bmInd){
			case 0: coeff = static_cast<int>(0x0000ffff); break; 
			case 1: coeff = static_cast<int>(0x0000ff01); break;
			case 2: coeff = static_cast<int>(0x000001ff); break;
			case 3: coeff = static_cast<int>(0x00000101); break;
		}
	}
};

template<>
__device__ int bmCalcInt<ChannelIn::SOFT4>(int i, encPack_t<ChannelIn::SOFT4>* coded, bmCalcHelper<ChannelIn::SOFT4>& helper){
	uint32_t data_raw = static_cast<uint32_t>(coded[i/4]);
	uint32_t mask0 = 0x000000f0 << (24-helper.shift);
	uint32_t mask1 = 0x0000000f << (24-helper.shift);
	uint32_t data0 = (data_raw & mask0) << (helper.shift);
	uint32_t data1 = (data_raw & mask1) << (helper.shift+4);
	int32_t data0shifted = static_cast<int32_t>(data0) >> 12;
	int32_t data1shifted = static_cast<int32_t>(data1) >> 28;
	int32_t data = (data0shifted&0xffff0000) | (data1shifted&0x0000ffff);
	return (__dp2a_lo(data, helper.coeff, 0));
}

//============================= SOFT8 ============================

template<>
struct bmCalcHelper<ChannelIn::SOFT8>{
	int coeff;
	int dataInd;
	int bmInd;
	__device__ bmCalcHelper(){
		dataInd = tx >> 2;
		bmInd = tx & 3;
		int shift = (dataInd&1)?0:16;
		switch(bmInd){
			case 0: coeff = static_cast<int>(0x0000ffff << shift); break; 
			case 1: coeff = static_cast<int>(0x0000ff01 << shift); break;
			case 2: coeff = static_cast<int>(0x000001ff << shift); break;
			case 3: coeff = static_cast<int>(0x00000101 << shift); break;
		}
	}
};

template<>
__device__ int bmCalcInt<ChannelIn::SOFT8>(int i, encPack_t<ChannelIn::SOFT8>* coded, bmCalcHelper<ChannelIn::SOFT8>& helper){
	return __dp4a(coded[i/2], helper.coeff, 0);
}

//============================= SOFT16 ============================

template<>
struct bmCalcHelper<ChannelIn::SOFT16>{
	int coeff;
	int dataInd;
	int bmInd;
	__device__ bmCalcHelper(){
		dataInd = tx >> 2;
		bmInd = tx & 3;
		switch(bmInd){
			case 0: coeff = static_cast<int>(0x0000ffff); break; //+0.5 in Q14
			case 1: coeff = static_cast<int>(0x0000ff01); break; //-0.5 in Q14
			case 2: coeff = static_cast<int>(0x000001ff); break; //-0.5 in Q14
			case 3: coeff = static_cast<int>(0x00000101); break; //+0.5 in Q14
		}
	}
};

template<>
__device__ int bmCalcInt<ChannelIn::SOFT16>(int i, encPack_t<ChannelIn::SOFT16>* coded, bmCalcHelper<ChannelIn::SOFT16>& helper){
	return __dp2a_lo(coded[i], helper.coeff, 0);
}

//============================= FP32 ============================

template<>
struct bmCalcHelper<ChannelIn::FP32>{
	float coeff0;
	float coeff1;
	float minVal;
	float maxVal;
	int dataInd;
	int bmInd;
	__device__ bmCalcHelper(){
		coeff0 = (tx & 2) ? 1.0f : -1.0f;
		coeff1 = (tx & 1) ? 1.0f : -1.0f;
		minVal = static_cast<float>(-(1<<(FPprecision-1)));
		maxVal = static_cast<float>((1<<(FPprecision-1))-1);
		dataInd = tx >> 2;
		bmInd = tx & 3;
	}
};

template<>
__device__ int bmCalcInt<ChannelIn::FP32>(int i, encPack_t<ChannelIn::FP32>* coded, bmCalcHelper<ChannelIn::FP32>& helper){
	encPack_t<ChannelIn::FP32> bit0 = coded[2*i];
	encPack_t<ChannelIn::FP32> bit1 = coded[2*i+1];
	bit0 = fminf(fmaxf(bit0, helper.minVal), helper.maxVal);
	bit1 = fminf(fmaxf(bit1, helper.minVal), helper.maxVal);
	return static_cast<int>((helper.coeff0 * bit0) + (helper.coeff1 * bit1));
}

//============================= BM Calculator ============================

template<Metric metricType>
__device__ metric_t<metricType> intToBM(int val){
	return val;
}

template<>
__device__ metric_t<Metric::M_B16> intToBM<Metric::M_B16>(int val){
	return static_cast<metric_t<Metric::M_B16>>(val);
}

template<>
__device__ metric_t<Metric::M_B32> intToBM<Metric::M_B32>(int val){
	return static_cast<metric_t<Metric::M_B32>>(val);
}

template<>
__device__ metric_t<Metric::M_FP16> intToBM<Metric::M_FP16>(int val){
	return __half(val);
}


template<ChannelIn inputType, Metric metricType>
__device__ void bmCalc(int stage, int num, metric_t<metricType> branchMetric[][4], encPack_t<inputType>* coded, bmCalcHelper<inputType>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		branchMetric[i%bmMemWidth][helper.bmInd] = intToBM<metricType>(bmCalcInt<inputType>(i, coded, helper));
		// if(bx==0 && ty==0)
		// 	printf("=== stage:%d ind:%d bm:%d\n", i, helper.bmInd, branchMetric[i%bmMemWidth][helper.bmInd]);	
	}
}

//-----------------------------------------------------------------------------

__device__ void bmIndCalc(unsigned int& allBmInd0, unsigned int& allBmInd1){
	allBmInd0 = 0;
	allBmInd1 = 0;
	for(int ind=CL-2; ind>=0; ind--){
		unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> ind;
		inState0 &= (1<<CL)-1;
		unsigned int inState1 = inState0 ^ (1<<(CL-1-ind));

		bool out0 = __popc(inState0 & ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::polyn1) % 2;
		bool out1 = __popc(inState0 & ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::polyn2) % 2;
		int bmInd = (out0 << 1) | out1;
		allBmInd0 = (allBmInd0 << 2) | bmInd;

		out0 = __popc(inState1 & ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::polyn1) % 2;
		out1 = __popc(inState1 & ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::polyn2) % 2;
		bmInd = (out0 << 1) | out1;
		allBmInd1 = (allBmInd1 << 2) | bmInd;
	}
}