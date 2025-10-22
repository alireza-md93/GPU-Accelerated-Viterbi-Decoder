#pragma once

//=====================================> GPU parameters <====================================//
//these parameters are used by viterbi core
//changing these parameters affects viterbi functionality
//if you do not know the algorithm, do not change
#define SHFTL_RAW 32 //shift_left:the length of left overlapping part of this block with the lefthand block  (without puncturing)
#define SHFTR_RAW 32 //shift_right:the length of right overlapping part of this block with the righthand block  (without puncturing)
#define D_RAW (256) //the total length of each block
//===========================================================================================//

#define CL 7 //constraint length
#define POLYN1 0171 //polynomial 1
#define POLYN2 0133 //polynomial 2

#define ROUNDUP(a, b) ( (a)<=0 ? 0 : (  ( (((a)-1)/(b)) + 1 ) * (b) ) ) //round up a to the nearest multiple of b

typedef int int32_t;
typedef short int int16_t;
typedef unsigned int uint32_t;
typedef unsigned short int uint16_t;    

enum Metric {B16=16, B32=32};

template<Metric metricType>
struct packType_helper;

template<> struct packType_helper<Metric::B16> { using type = uint16_t; };
template<> struct packType_helper<Metric::B32> { using type = uint32_t; };

template<Metric metricType>
using pack_t = typename packType_helper<metricType>::type;

template<Metric metricType>
struct metricType_helper;

template<> struct metricType_helper<Metric::B16> { using type = int16_t; };
template<> struct metricType_helper<Metric::B32> { using type = int32_t; };

template<Metric metricType>
using metric = typename metricType_helper<metricType>::type;


//=====================================> GPU parameters <====================================//
#define SHFTL (ROUNDUP(SHFTL_RAW, 32) - (CL-1))
#define SHFTR (ROUNDUP(SHFTR_RAW, 32) + (CL-1))
#define DECSIZE  ROUNDUP(D_RAW, 32)
#define SLIDESIZE 32
#define SHMEMWIDTH (SHFTL + SLIDESIZE + SHFTR)
#define BLOCK_DIMY 2
//===========================================================================================//
