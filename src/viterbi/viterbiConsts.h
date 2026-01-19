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

constexpr int CL = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::constLen;
constexpr int extraL = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::extraL;
constexpr int extraR = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::extraR;
constexpr int slideSize = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::slideSize;
constexpr int forwardLen = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::forwardLen;
constexpr int bmMemWidth = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::bmMemWidth;
constexpr int FPprecision = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::FPprecision;
template<DecodeOut outputType> constexpr int bpp = ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, outputType, CompMode::REG>::bitsPerPack;
template<Metric metricType> constexpr int bpm = ViterbiCUDA<ChannelIn::HARD, metricType, DecodeOut::O_B16, CompMode::REG>::bitsPerMetric;
template<ChannelIn inputType> constexpr int dpp = ViterbiCUDA<inputType, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::encDataPerPack;
template<ChannelIn inputType> constexpr int chnWidth = ViterbiCUDA<inputType, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::encDataWidth;

template<Metric metricType> using metric_t = typename ViterbiCUDA<ChannelIn::HARD, metricType, DecodeOut::O_B16, CompMode::REG>::metric_t;
template<DecodeOut outputType> using decPack_t = typename ViterbiCUDA<ChannelIn::HARD, Metric::M_B16, outputType, CompMode::REG>::decPack_t;
template<ChannelIn inputType> using encPack_t = typename ViterbiCUDA<inputType, Metric::M_B16, DecodeOut::O_B16, CompMode::REG>::encPack_t;