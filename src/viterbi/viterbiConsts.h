#pragma once

#include "viterbi.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define bdx blockDim.x
#define bdy blockDim.y
#define gdx gridDim.x
#define gdy gridDim.y

constexpr int CL = ViterbiCUDA<>::constLen;
constexpr int extraL = ViterbiCUDA<>::extraL;
constexpr int extraR = ViterbiCUDA<>::extraR;
constexpr int slideSize = ViterbiCUDA<>::slideSize;
constexpr int forwardLen = ViterbiCUDA<>::forwardLen;
constexpr int bmMemWidth = ViterbiCUDA<>::bmMemWidth;
constexpr int FPprecision = ViterbiCUDA<>::FPprecision;
template<DecodeOut outputType> constexpr int bpp = ViterbiCUDA<outputType>::bitsPerPack;
template<Metric metricType> constexpr int bpm = ViterbiCUDA<metricType>::bitsPerMetric;
template<ChannelIn inputType> constexpr int dpp = ViterbiCUDA<inputType>::encDataPerPack;
template<ChannelIn inputType> constexpr int chnWidth = ViterbiCUDA<inputType>::encDataWidth;

template<Metric metricType> using metric_t = typename ViterbiCUDA<metricType>::metric_t;
template<DecodeOut outputType> using decPack_t = typename ViterbiCUDA<outputType>::decPack_t;
template<ChannelIn inputType> using encPack_t = typename ViterbiCUDA<inputType>::encPack_t;