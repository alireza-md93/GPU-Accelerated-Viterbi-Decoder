#pragma once

#include "viterbi.h"
#include "viterbiConsts.h"
#include <cstdint>

template<ChannelIn inputType>
struct bmCalcHelper;

template<Metric metricType, ChannelIn inputType>
__device__ void bmCalc(int stage, int num, metric_t<metricType> branchMetric[][4], encPack_t<inputType>* coded, bmCalcHelper<inputType>& helper){}

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
__device__ void bmCalc<Metric::B16, ChannelIn::HARD>
(int stage, int num, metric_t<Metric::B16> branchMetric[][4], encPack_t<ChannelIn::HARD>* coded, bmCalcHelper<ChannelIn::HARD>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>1)){
		uint32_t data = coded[i>>4];
		branchMetric[i%shmemWidth][helper.bmInd] = 1 - static_cast<metric_t<Metric::B16>>(__popc((data&helper.mask0) ^ helper.out0));
		branchMetric[(i+(bdx>>2))%shmemWidth][helper.bmInd] = 1 - static_cast<metric_t<Metric::B16>>(__popc((data&helper.mask1) ^ helper.out1));
	}
}

template<>
__device__ void bmCalc<Metric::B32, ChannelIn::HARD>
(int stage, int num, metric_t<Metric::B32> branchMetric[][4], encPack_t<ChannelIn::HARD>* coded, bmCalcHelper<ChannelIn::HARD>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>1)){
		uint32_t data = coded[i>>4];
		branchMetric[i%shmemWidth][helper.bmInd] = 1 - __popc((data&helper.mask0) ^ helper.out0);
		branchMetric[(i+(bdx>>2))%shmemWidth][helper.bmInd] = 1 - __popc((data&helper.mask1) ^ helper.out1);
	}
}

template<>
__device__ void bmCalc<Metric::FP16, ChannelIn::HARD>
(int stage, int num, metric_t<Metric::FP16> branchMetric[][4], encPack_t<ChannelIn::HARD>* coded, bmCalcHelper<ChannelIn::HARD>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>1)){
		uint32_t data = coded[i>>4];
		branchMetric[i%shmemWidth][helper.bmInd] = __half(1 - __popc((data&helper.mask0) ^ helper.out0));
		branchMetric[(i+(bdx>>2))%shmemWidth][helper.bmInd] = __half(1 - __popc((data&helper.mask1) ^ helper.out1));
	}
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
__device__ void bmCalc<Metric::B16, ChannelIn::SOFT4>
(int stage, int num, metric_t<Metric::B16> branchMetric[][4], encPack_t<ChannelIn::SOFT4>* coded, bmCalcHelper<ChannelIn::SOFT4>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		uint32_t data_raw = static_cast<uint32_t>(coded[i/4]);
		uint32_t mask0 = 0x000000f0 << (24-helper.shift);
		uint32_t mask1 = 0x0000000f << (24-helper.shift);
		uint32_t data0 = (data_raw & mask0) << (helper.shift);
		uint32_t data1 = (data_raw & mask1) << (helper.shift+4);
		int32_t data0shifted = static_cast<int32_t>(data0) >> 12;
		int32_t data1shifted = static_cast<int32_t>(data1) >> 28;
		int32_t data = (data0shifted&0xffff0000) | (data1shifted&0x0000ffff);

		branchMetric[i%shmemWidth][helper.bmInd] = static_cast<metric_t<Metric::B16>>(__dp2a_lo(data, helper.coeff, 0));
	}
}

template<>
__device__ void bmCalc<Metric::B32, ChannelIn::SOFT4>
(int stage, int num, metric_t<Metric::B32> branchMetric[][4], encPack_t<ChannelIn::SOFT4>* coded, bmCalcHelper<ChannelIn::SOFT4>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		uint32_t data_raw = static_cast<uint32_t>(coded[i/4]);
		uint32_t mask0 = 0x000000f0 << (24-helper.shift);
		uint32_t mask1 = 0x0000000f << (24-helper.shift);
		uint32_t data0 = (data_raw & mask0) << (helper.shift);
		uint32_t data1 = (data_raw & mask1) << (helper.shift+4);
		int32_t data0shifted = static_cast<int32_t>(data0) >> 12;
		int32_t data1shifted = static_cast<int32_t>(data1) >> 28;
		int32_t data = (data0shifted&0xffff0000) | (data1shifted&0x0000ffff);

		branchMetric[i%shmemWidth][helper.bmInd] = __dp2a_lo(data, helper.coeff, 0);
	}
}

template<>
__device__ void bmCalc<Metric::FP16, ChannelIn::SOFT4>
(int stage, int num, metric_t<Metric::FP16> branchMetric[][4], encPack_t<ChannelIn::SOFT4>* coded, bmCalcHelper<ChannelIn::SOFT4>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		uint32_t data_raw = static_cast<uint32_t>(coded[i/4]);
		uint32_t mask0 = 0x000000f0 << (24-helper.shift);
		uint32_t mask1 = 0x0000000f << (24-helper.shift);
		uint32_t data0 = (data_raw & mask0) << (helper.shift);
		uint32_t data1 = (data_raw & mask1) << (helper.shift+4);
		int32_t data0shifted = static_cast<int32_t>(data0) >> 12;
		int32_t data1shifted = static_cast<int32_t>(data1) >> 28;
		int32_t data = (data0shifted&0xffff0000) | (data1shifted&0x0000ffff);

		branchMetric[i%shmemWidth][helper.bmInd] = __half(__dp2a_lo(data, helper.coeff, 0));
	}
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
__device__ void bmCalc<Metric::B16, ChannelIn::SOFT8>
(int stage, int num, metric_t<Metric::B16> branchMetric[][4], encPack_t<ChannelIn::SOFT8>* coded, bmCalcHelper<ChannelIn::SOFT8>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = static_cast<metric_t<Metric::B16>>(__dp4a(coded[i/2], helper.coeff, 0));
}

template<>
__device__ void bmCalc<Metric::B32, ChannelIn::SOFT8>
(int stage, int num, metric_t<Metric::B32> branchMetric[][4], encPack_t<ChannelIn::SOFT8>* coded, bmCalcHelper<ChannelIn::SOFT8>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = __dp4a(coded[i/2], helper.coeff, 0);
}

template<>
__device__ void bmCalc<Metric::FP16, ChannelIn::SOFT8>
(int stage, int num, metric_t<Metric::FP16> branchMetric[][4], encPack_t<ChannelIn::SOFT8>* coded, bmCalcHelper<ChannelIn::SOFT8>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = __half(__dp4a(coded[i/2], helper.coeff, 0));
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
__device__ void bmCalc<Metric::B16, ChannelIn::SOFT16>
(int stage, int num, metric_t<Metric::B16> branchMetric[][4], encPack_t<ChannelIn::SOFT16>* coded, bmCalcHelper<ChannelIn::SOFT16>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = static_cast<metric_t<Metric::B16>>(__dp2a_lo(coded[i], helper.coeff, 0));
}

template<>
__device__ void bmCalc<Metric::B32, ChannelIn::SOFT16>
(int stage, int num, metric_t<Metric::B32> branchMetric[][4], encPack_t<ChannelIn::SOFT16>* coded, bmCalcHelper<ChannelIn::SOFT16>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = __dp2a_lo(coded[i], helper.coeff, 0); 
}

template<>
__device__ void bmCalc<Metric::FP16, ChannelIn::SOFT16>
(int stage, int num, metric_t<Metric::FP16> branchMetric[][4], encPack_t<ChannelIn::SOFT16>* coded, bmCalcHelper<ChannelIn::SOFT16>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2))
		branchMetric[i%shmemWidth][helper.bmInd] = __half(__dp2a_lo(coded[i], helper.coeff, 0));
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
__device__ void bmCalc<Metric::B16, ChannelIn::FP32>
(int stage, int num, metric_t<Metric::B16> branchMetric[][4], encPack_t<ChannelIn::FP32>* coded, bmCalcHelper<ChannelIn::FP32>& helper){

	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		encPack_t<ChannelIn::FP32> bit0 = coded[2*i];
		encPack_t<ChannelIn::FP32> bit1 = coded[2*i+1];
		bit0 = fminf(fmaxf(bit0, helper.minVal), helper.maxVal);
		bit1 = fminf(fmaxf(bit1, helper.minVal), helper.maxVal);
		branchMetric[i%shmemWidth][helper.bmInd] = static_cast<metric_t<Metric::B16>>((helper.coeff0 * bit0) + (helper.coeff1 * bit1));
	}
}

template<>
__device__ void bmCalc<Metric::B32, ChannelIn::FP32>
(int stage, int num, metric_t<Metric::B32> branchMetric[][4], encPack_t<ChannelIn::FP32>* coded, bmCalcHelper<ChannelIn::FP32>& helper){
	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		encPack_t<ChannelIn::FP32> bit0 = coded[2*i];
		encPack_t<ChannelIn::FP32> bit1 = coded[2*i+1];
		bit0 = fminf(fmaxf(bit0, helper.minVal), helper.maxVal);
		bit1 = fminf(fmaxf(bit1, helper.minVal), helper.maxVal);
		branchMetric[i%shmemWidth][helper.bmInd] = static_cast<metric_t<Metric::B32>>((helper.coeff0 * bit0) + (helper.coeff1 * bit1)); 
	}
}

template<>
__device__ void bmCalc<Metric::FP16, ChannelIn::FP32>
(int stage, int num, metric_t<Metric::FP16> branchMetric[][4], encPack_t<ChannelIn::FP32>* coded, bmCalcHelper<ChannelIn::FP32>& helper){

	for(int i=stage+helper.dataInd; i<stage+num; i+=(bdx>>2)){
		encPack_t<ChannelIn::FP32> bit0 = coded[2*i];
		encPack_t<ChannelIn::FP32> bit1 = coded[2*i+1];
		bit0 = fminf(fmaxf(bit0, helper.minVal), helper.maxVal);
		bit1 = fminf(fmaxf(bit1, helper.minVal), helper.maxVal);
		branchMetric[i%shmemWidth][helper.bmInd] = __half((helper.coeff0 * bit0) + (helper.coeff1 * bit1));
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

		bool out0 = __popc(inState0 & ViterbiCUDA<Metric::B16, ChannelIn::HARD>::polyn1) % 2;
		bool out1 = __popc(inState0 & ViterbiCUDA<Metric::B16, ChannelIn::HARD>::polyn2) % 2;
		int bmInd = (out0 << 1) | out1;
		allBmInd0 = (allBmInd0 << 2) | bmInd;

		out0 = __popc(inState1 & ViterbiCUDA<Metric::B16, ChannelIn::HARD>::polyn1) % 2;
		out1 = __popc(inState1 & ViterbiCUDA<Metric::B16, ChannelIn::HARD>::polyn2) % 2;
		bmInd = (out0 << 1) | out1;
		allBmInd1 = (allBmInd1 << 2) | bmInd;
	}
}