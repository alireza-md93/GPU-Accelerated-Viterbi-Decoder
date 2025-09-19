#include "cuda_runtime.h"
#include "gputimer.h"
#include <stdio.h>

#define HANDLE_ERROR(ans) if (ans != cudaSuccess) {printf( "%s in %s at line %d\n", cudaGetErrorString( ans ),__FILE__, __LINE__ );}
	//exit( EXIT_FAILURE );

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define bdx blockDim.x
#define gdx gridDim.x
#define gdy gridDim.y


//=====================================> GPU parameters <====================================//
//these parameters are used by viterbi core
//changing these parameters affects viterbi functionality
//if you do not know the algorithm, do not change
#define SHFTL_RAW 32 //shift_left:the length of left overlapping part of this block with the lefthand block  (without puncturing)
#define SHFTR_RAW 32 //shift_right:the length of right overlapping part of this block with the righthand block  (without puncturing)
#define D_RAW 256 //the total length of each block
//===========================================================================================//



//=====================================> coding parameters <====================================//
#define CL 7 //constraint length
#define POLYN1 0171 //polynomial 1
#define POLYN2 0133 //polynomial 2

#define PATHSIZE 8
typedef unsigned char path_t;

// #define PATHSIZE 32
// typedef unsigned int path_t;

#define TPB (1<<(CL-2))

#define ROUNDUP(a, b) ( (a)<=0 ? 0 : (  ( (((a)-1)/(b)) + 1 ) * (b) ) ) //round up a to the nearest multiple of b

//=====================================> GPU parameters <====================================//
#define SHFTL ROUNDUP(SHFTL_RAW, PATHSIZE)
#define SHFTR ROUNDUP(SHFTR_RAW, PATHSIZE)
#define DECSIZE  ROUNDUP(D_RAW, PATHSIZE)
#define SLIDESIZE 32
#define SHMEMWIDTH (SHFTL + SLIDESIZE + SHFTR)
//===========================================================================================//

__device__ __forceinline__ void bmCalc(int ind, int num, float branchMetric[][2], float* coded){
	for(int i=ind; i<ind+num; i+=bdx){
		branchMetric[(i+tx)%SHMEMWIDTH][0] = -coded[2*(i+tx)] - coded[2*(i+tx)+1];
		branchMetric[(i+tx)%SHMEMWIDTH][1] = -coded[2*(i+tx)] + coded[2*(i+tx)+1];
	}   
}

__device__ __forceinline__ void forwardACS(int ind, float* pathMetric, unsigned int pathPrev[][1<<(CL-1)], float branchMetric[][2], char ind_s0_b0, int prevState0, int prevState1){
	int i = ind % SHMEMWIDTH;
	float pathMetric1 = pathMetric[prevState0] + branchMetric[i][ind_s0_b0];
	float pathMetric2 = pathMetric[prevState1] - branchMetric[i][ind_s0_b0];
	float pathMetric3 = pathMetric[prevState0] - branchMetric[i][ind_s0_b0];
	float pathMetric4 = pathMetric[prevState1] + branchMetric[i][ind_s0_b0];
	
	bool condPM12 = pathMetric1 > pathMetric2;
	bool condPM34 = pathMetric3 > pathMetric4;
	
	bool condPrev = prevState0 % 2 == 0;
	
	bool cond1 = condPM12 ^ condPrev;
	bool cond2 = condPM34 ^ condPrev;
	
	pathPrev[i/32][tx] <<= 1;
	pathPrev[i/32][tx] += (char)(cond1);
	
	pathPrev[i/32][tx+(1<<(CL-2))] <<= 1;
	pathPrev[i/32][tx+(1<<(CL-2))] += (char)(cond2);
	
	pathMetric1 = condPM12 ? pathMetric1 : pathMetric2;
	pathMetric3 = condPM34 ? pathMetric3 : pathMetric4;
	
	__syncthreads();
	
	//update path metrics
	pathMetric[tx] = pathMetric1;
	pathMetric[tx+(1<<(CL-2))] = pathMetric3;
	
	__syncthreads();
}

__device__ __forceinline__ void traceback(int endStage, int dataEndInd, bool* data, float* pathMetric, unsigned int pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		//recognize the last state of survived path that is with the maximum path metric
		float winnerPathMetric = pathMetric[0];
		int winnerState = 0;
		for(int i=0; i<(1<<(CL-1)); i++)
			if(pathMetric[i] > winnerPathMetric){
				winnerPathMetric = pathMetric[i];
				winnerState = i;
			}
		bool insBit=winnerState&1;
		winnerState >>= 1;
	
	
		//part1. only trace back process to reach the second part
		for(int s=0; s<SHFTR; s++){
			int i = (endStage - s) % SHMEMWIDTH;
			winnerState = ((winnerState << 1) & ~(1<<(CL-1))) + (int)insBit;
			insBit = (pathPrev[i/32][winnerState] &(1U<<(31-i%32))) != 0;
		}
		
		for(int s=0; s<SLIDESIZE; s++){
			winnerState = ((winnerState << 1) & ~(1<<(CL-1))) + (int)insBit; //trace back
			data[dataEndInd-s] = winnerState>>(CL-2); //decode
			int i = (endStage - SHFTR - s) % SHMEMWIDTH;
			insBit = (pathPrev[i/32][winnerState] & (1U<<(31-i%32))) != 0;
		}
	}
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
__global__ void viterbi_core(bool* data, float* coded){
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	
	__shared__ float pathMetric[1<<(CL-1)]; //path metrics
	__shared__ unsigned int pathPrev[(SHMEMWIDTH-1)/32+1][1<<(CL-1)]; //survived paths
	__shared__ float branchMetric[SHMEMWIDTH][2];

	char ind_s0_b0;
	int prevState0, prevState1; //previous states
	
	//shift "data" and "coded" pointer to the index that this block should process
	int start_ind = DECSIZE * bx;
	data += start_ind;
	coded += start_ind*2;
	
	/***************************** calculate output matrix of FSM *****************************/	 
	
	bool out0[2];

	//the first output
	prevState0 = (tx<<1);
	
	out0[0] = __popc(prevState0 & POLYN1) % 2;
	out0[1] = __popc(prevState0 & POLYN2) % 2;

	prevState0 = (tx<<1) + (int)(out0[0]); 
	prevState1 = (tx<<1) + (int)(!out0[0]);
	
	ind_s0_b0 = out0[0] ^ out0[1];

	/******************************************************************************************/
	
	//initialize path metrics
	//each thread initializes one element
	pathMetric[tx] = 0.0;
	pathMetric[tx+(1<<(CL-2))] = 0.0;	
	
	bmCalc(0, SHFTL+SHFTR, branchMetric, coded);
	__syncthreads();
	for(int i=0; i<SHFTL+SHFTR; i++)
		forwardACS(i, pathMetric, pathPrev, branchMetric, ind_s0_b0, prevState0, prevState1);

	for(int slide=0; slide<DECSIZE; slide+=SLIDESIZE){
		int ind = slide + SHFTL + SHFTR;
		bmCalc(ind, SLIDESIZE, branchMetric, coded);   
		__syncthreads();
		for(int i=ind; i<ind+SLIDESIZE; i++){	
			forwardACS(i, pathMetric, pathPrev, branchMetric, ind_s0_b0, prevState0, prevState1);
		}
		traceback(ind+SLIDESIZE-1, slide+SLIDESIZE-1, data, pathMetric, pathPrev);
	}
}

//-----------------------------------------------------------------------------

int viterbi_run(float* input_d, bool* output_d, int messageLen, float* time) {
	int wins = messageLen / DECSIZE;

	GpuTimer timer;
	
	dim3 grid (wins, 1, 1); 
	dim3 block (TPB, 1, 1);

	timer.Start();
    viterbi_core <<<grid, block>>> (output_d, input_d);
	timer.Stop();
	*time = timer.Elapsed();
	
	HANDLE_ERROR(   cudaPeekAtLastError()   );
	
	return 1;
}
