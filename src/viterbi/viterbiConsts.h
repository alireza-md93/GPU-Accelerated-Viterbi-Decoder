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

constexpr int CL = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::constLen;
constexpr int extraL = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::extraL;
constexpr int extraR = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::extraR;
constexpr int slideSize = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::slideSize;
constexpr int shmemWidth = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::shMemWidth;
constexpr int FPprecision = ViterbiCUDA<Metric::B16, ChannelIn::HARD>::FPprecision;
template<Metric metricType> constexpr int bpp = ViterbiCUDA<metricType, ChannelIn::HARD>::bitsPerPack;
template<ChannelIn inputType> constexpr int dpp = ViterbiCUDA<Metric::B16, inputType>::encDataPerPack;
template<ChannelIn inputType> constexpr int chnWidth = ViterbiCUDA<Metric::B16, inputType>::encDataWidth;

template<Metric metricType> using metric_t = typename ViterbiCUDA<metricType, ChannelIn::HARD>::metric_t;
template<Metric metricType> using decPack_t = typename ViterbiCUDA<metricType, ChannelIn::HARD>::decPack_t;
template<ChannelIn inputType> using encPack_t = typename ViterbiCUDA<Metric::B16, inputType>::encPack_t;