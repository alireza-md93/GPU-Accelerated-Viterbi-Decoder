#include "cuda_runtime.h"
#include "gputimer.h"
#include "gpuerrors.h"
#include "parameters.h"
#include <stdio.h>

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

__device__ __forceinline__ void bmCalc(int ind, int num, float branchMetric[][2], float* coded){
	for(int i=ind+tx; i<ind+num; i+=bdx){
		branchMetric[i%SHMEMWIDTH][0] = -coded[2*i] - coded[2*i+1];
		branchMetric[i%SHMEMWIDTH][1] = -coded[2*i] + coded[2*i+1];
	}   
}

template<ACS acsType>
__device__ __forceinline__ void forwardACS(int ind, float* pathMetric, path_t pathPrev[][1<<(CL-1)], float branchMetric[][2], char bmInd, int prevState, bool out0){}

template<>
__device__ __forceinline__ void forwardACS<ACS::RADIX2>(int ind, float* pathMetric, path_t pathPrev[][1<<(CL-1)], float branchMetric[][2], char bmInd, int prevState, bool out0){
	int i = ind % SHMEMWIDTH;

	float bm = out0 ? -branchMetric[i][bmInd] : branchMetric[i][bmInd];

	float pathMetric1 = pathMetric[prevState]   + bm;
	float pathMetric2 = pathMetric[prevState+1] - bm;
	float pathMetric3 = pathMetric[prevState]   - bm;
	float pathMetric4 = pathMetric[prevState+1] + bm;
	
	bool condPM12 = pathMetric1 < pathMetric2;
	bool condPM34 = pathMetric3 < pathMetric4;
	
	pathPrev[i/PATHSIZE][tx] <<= 1;
	pathPrev[i/PATHSIZE][tx] += condPM12;
	
	pathPrev[i/PATHSIZE][tx+(1<<(CL-2))] <<= 1;
	pathPrev[i/PATHSIZE][tx+(1<<(CL-2))] += condPM34;
	
	pathMetric1 = condPM12 ? pathMetric2 : pathMetric1;
	pathMetric3 = condPM34 ? pathMetric4 : pathMetric3;
	
	__syncthreads();
	
	//update path metrics
	pathMetric[tx] = pathMetric1;
	pathMetric[tx+(1<<(CL-2))] = pathMetric3;
	
	__syncthreads();
}

template<>
__device__ __forceinline__ void forwardACS<ACS::SIMPLE>(int ind, float* pathMetric, path_t pathPrev[][1<<(CL-1)], float branchMetric[][2], char bmInd, int prevState, bool out0){
	int i = ind % SHMEMWIDTH;

	float bm = out0 ? -branchMetric[i][bmInd] : branchMetric[i][bmInd];

	float pathMetric1 = pathMetric[prevState]   + bm;
	float pathMetric2 = pathMetric[prevState+1] - bm;
	
	bool condPM12 = pathMetric1 < pathMetric2;
	
	pathPrev[i/PATHSIZE][tx] <<= 1;
	pathPrev[i/PATHSIZE][tx] += condPM12;
	
	pathMetric1 = condPM12 ? pathMetric2 : pathMetric1;
	
	__syncthreads();
	
	//update path metrics
	pathMetric[tx] = pathMetric1;
	
	__syncthreads();
}

__device__ __forceinline__ void traceback(int endStage, int dataEndInd, bool* data, float* pathMetric, path_t pathPrev[][1<<(CL-1)]){
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
			insBit = (pathPrev[i/PATHSIZE][winnerState] &(1U<<((PATHSIZE-1)-i%PATHSIZE))) != 0;
		}
		
		for(int s=0; s<SLIDESIZE; s++){
			winnerState = ((winnerState << 1) & ~(1<<(CL-1))) + (int)insBit; //trace back
			data[dataEndInd-s] = winnerState>>(CL-2); //decode
			int i = (endStage - SHFTR - s) % SHMEMWIDTH;
			insBit = (pathPrev[i/PATHSIZE][winnerState] & (1U<<((PATHSIZE-1)-i%PATHSIZE))) != 0;
		}
	}
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<ACS ACStype>
__global__ void viterbi_core(bool* data, float* coded, path_t* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ float sharedMem[];
	float* sharedMemTip = sharedMem;
	sharedMemTip += ty * ((1<<(CL-1)) + (SHMEMWIDTH * 2));
	float* pathMetric = sharedMemTip;
	sharedMemTip += (1<<(CL-1));
	float (*branchMetric)[2] = (float(*)[2])sharedMemTip;

	path_t (*pathPrev) [1<<(CL-1)] = (path_t (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) *((SHMEMWIDTH-1)/PATHSIZE+1);

	int bmInd, prevState;
	
	//shift "data" and "coded" pointer to the index that this block should process
	int start_ind = DECSIZE * bdy * bx + DECSIZE * ty;
	data += start_ind;
	coded += start_ind*2;
	
	/****************************** calculate trellis parameters ******************************/	 
	bool out0 = __popc((tx<<1) & POLYN1) % 2;
	bool out1 = __popc((tx<<1) & POLYN2) % 2;
	prevState = ((tx<<1) & ((1<<(CL-1)) - 1)); 
	bmInd = out0 ^ out1;
	/******************************************************************************************/
	
	//initialize path metrics
	//each thread initializes one element
	pathMetric[tx] = 0.0;
	pathMetric[tx+(1<<(CL-2))] = 0.0;	
	
	bmCalc(0, SHFTL+SHFTR, branchMetric, coded);
	__syncthreads();
	for(int i=0; i<SHFTL+SHFTR; i++)
		forwardACS<ACStype>(i, pathMetric, pathPrev, branchMetric, bmInd, prevState, out0);

	for(int slide=0; slide<DECSIZE; slide+=SLIDESIZE){
		int ind = slide + SHFTL + SHFTR;
		bmCalc(ind, SLIDESIZE, branchMetric, coded);   
		__syncthreads();
		for(int i=ind; i<ind+SLIDESIZE; i++){	
			forwardACS<ACStype>(i, pathMetric, pathPrev, branchMetric, bmInd, prevState, out0);
		}
		traceback(ind+SLIDESIZE-1, slide+SLIDESIZE-1, data, pathMetric, pathPrev);
	}
}

//-----------------------------------------------------------------------------

int viterbi_run(float* input_d, bool* output_d, int messageLen, float* time, ACS acsType) {
	int wins = messageLen / DECSIZE;
	path_t* pathPrev_d;
	int ppSize = ((SHMEMWIDTH-1)/PATHSIZE+1) * (1<<(CL-1)) * sizeof(path_t) * wins;
	int sharedMemSize = (1<<(CL-1)) * sizeof(float);
	sharedMemSize += SHMEMWIDTH * 2 * sizeof(float);
	sharedMemSize *= BLOCK_DIMY;
	int blockSize = (acsType == ACS::SIMPLE) ? (1<<(CL-1)) : (1<<(CL-2));

	HANDLE_ERROR(cudaMalloc((void**)&pathPrev_d, ppSize));

	GpuTimer timer;
	
	dim3 grid (wins/BLOCK_DIMY, 1, 1); 
	dim3 block (blockSize, BLOCK_DIMY, 1);

	timer.Start();
	switch (acsType)
	{
	case ACS::SIMPLE:
		viterbi_core<ACS::SIMPLE> <<<grid, block, sharedMemSize>>> (output_d, input_d, pathPrev_d);
		break;
	case ACS::RADIX2:
		viterbi_core<ACS::RADIX2> <<<grid, block, sharedMemSize>>> (output_d, input_d, pathPrev_d);
		break;
	
	default:
		break;
	}
	timer.Stop();
	*time = timer.Elapsed();
	
	HANDLE_ERROR(   cudaPeekAtLastError()   );
	HANDLE_ERROR(cudaFree(pathPrev_d));
	
	return 1;
}
