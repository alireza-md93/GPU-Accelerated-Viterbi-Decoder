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

#define TPB (1<<(CL-2))

#define ROUNDUP(a, b) ( (a)<=0 ? 0 : (  ( (((a)-1)/(b)) + 1 ) * (b) ) ) //round up a to the nearest multiple of b

#define PATHSIZE 8
typedef unsigned char path_t;

// #define PATHSIZE 32
// typedef unsigned int path_t;

//=====================================> GPU parameters <====================================//
#define SHFTL ROUNDUP(SHFTL_RAW, PATHSIZE)
#define SHFTR ROUNDUP(SHFTR_RAW, PATHSIZE)
#define DECSIZE  ROUNDUP(D_RAW, PATHSIZE)
#define SLIDESIZE 32
#define SHMEMWIDTH (SHFTL + SLIDESIZE + SHFTR)
#define BLOCK_DIMY 2
//===========================================================================================//
