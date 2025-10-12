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

struct trellis_t{
	float pm0;
	float pm1;
	path_t pp0;
	path_t pp1;
};

__device__ __forceinline__ void bmCalc(int ind, int num, float branchMetric[][2], float* coded){
	for(int i=ind+tx; i<ind+num; i+=bdx){
		// if(bx==0 && ty==0)printf("=== out:%d:%d%d\n", i, coded[2*i]>0.0, coded[2*i+1]>0.0);
		branchMetric[i%SHMEMWIDTH][0] = -coded[2*i] - coded[2*i+1];
		branchMetric[i%SHMEMWIDTH][1] = -coded[2*i] + coded[2*i+1];
	}   
}

template<ACS acsType>
__device__ __forceinline__ void forwardACS(int ind, trellis_t* old, trellis_t* now, path_t pathPrev[][1<<(CL-1)], float branchMetric[][2]){}

template<>
__device__ __forceinline__ void forwardACS<ACS::RADIX2>(int ind, trellis_t* old, trellis_t* now, path_t pathPrev[][1<<(CL-1)], float branchMetric[][2]){
	int i = ind % SHMEMWIDTH;

	//input bit plus state
	unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> (ind%(CL-1));
	inState0 &= (1<<CL)-1;
	unsigned int inState1 = inState0 ^ (1<<(CL-1-ind%(CL-1)));

	if(i%(CL-1) < CL-6){
		bool out0 = __popc(inState0 & POLYN1) % 2;
		bool out1 = __popc(inState0 & POLYN2) % 2;
		int bmInd = out0 ^ out1;

		float bm = out0 ? -branchMetric[i][bmInd] : branchMetric[i][bmInd];

		float pm0_0 = old->pm0 + bm;
		float pm1_0 = old->pm1 - bm;
		bool cond0 = pm0_0 < pm1_0;
		now->pp0 = cond0 ? old->pp1 : old->pp0;
		now->pp0 = (now->pp0 << 1) + cond0;
		now->pm0 = cond0 ? pm1_0 : pm0_0;

		float pm0_1 = old->pm0 - bm;
		float pm1_1 = old->pm1 + bm;
		bool cond1 = pm0_1 < pm1_1;
		now->pp1 = cond1 ? old->pp1 : old->pp0;
		now->pp1 = (now->pp1 << 1) + cond1;
		now->pm1 = cond1 ? pm1_1 : pm0_1;

		*old = *now;
	}
	else{
		now->pm0 = __shfl_xor_sync(0xffffffff, now->pm0, 1<<(ind%(CL-1)-CL+6));
		now->pm1 = __shfl_xor_sync(0xffffffff, now->pm1, 1<<(ind%(CL-1)-CL+6));
		now->pp0 = __shfl_xor_sync(0xffffffff, now->pp0, 1<<(ind%(CL-1)-CL+6));
		now->pp1 = __shfl_xor_sync(0xffffffff, now->pp1, 1<<(ind%(CL-1)-CL+6));
		
		bool out0 = __popc(inState0 & POLYN1) % 2;
		bool out1 = __popc(inState0 & POLYN2) % 2;
		int bmInd = out0 ^ out1;

		float bm = out0 ? -branchMetric[i][bmInd] : branchMetric[i][bmInd];

		float pmFromThis = old->pm0 + bm;
		float pmFromThat = now->pm0 - bm;
		bool cond0 = pmFromThis < pmFromThat;
		bool bit0 = (inState0 & 1) ^ cond0;
		now->pp0 = cond0 ? now->pp0 : old->pp0;
		now->pp0 = (now->pp0 << 1) + bit0;
		now->pm0 = cond0 ? pmFromThat : pmFromThis;

		out0 = __popc(inState1 & POLYN1) % 2;
		out1 = __popc(inState1 & POLYN2) % 2;
		bmInd = out0 ^ out1;

		bm = out0 ? -branchMetric[i][bmInd] : branchMetric[i][bmInd];

		pmFromThis = old->pm1 + bm;
		pmFromThat = now->pm1 - bm;
		bool cond1 = pmFromThis < pmFromThat;
		bool bit1 = (inState1 & 1) ^ cond1;
		now->pp1 = cond1 ? now->pp1 : old->pp1;
		now->pp1 = (now->pp1 << 1) + bit1;
		now->pm1 = cond1 ? pmFromThat : pmFromThis;

		*old = *now;
	}

	if((i+1) % PATHSIZE == 0){
		pathPrev[i/PATHSIZE][inState0>>1] = now->pp0;
		pathPrev[i/PATHSIZE][inState1>>1] = now->pp1;
	}

	// if(bx==0 && ty==0)printf("=== stage:%d state:%d pm:%f pp:%u\n", ind, inState0>>1, now->pm0, (now->pp0) & 1);
	// if(bx==0 && ty==0)printf("=== stage:%d state:%d pm:%f pp:%u\n", ind, inState1>>1, now->pm1, (now->pp1) & 1);
	// if(bx==0 && ty==0 && (i+1)%PATHSIZE==0)printf("=== stage:%d state:%d PP:%u\n", ind, inState0>>1, now->pp0);
	// if(bx==0 && ty==0 && (i+1)%PATHSIZE==0)printf("=== stage:%d state:%d PP:%u\n", ind, inState1>>1, now->pp1);
}

__device__ __forceinline__ void traceback(int endStage, int dataEndInd, path_t* data, float* pathMetric, path_t pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		//recognize the last state of survived path that is with the maximum path metric
		// float winnerPathMetric = pathMetric[0];
		int winnerState = 0;
		// for(int i=0; i<(1<<(CL-1)); i++)
		// 	if(pathMetric[i] > winnerPathMetric){
		// 		winnerPathMetric = pathMetric[i];
		// 		winnerState = i;
		// 	}
	
		for(int s=0; s<SHFTR-PATHSIZE; s+=PATHSIZE){
			path_t pp = pathPrev[((endStage - s) % SHMEMWIDTH)/PATHSIZE][winnerState];
			winnerState = __brev(pp<<(32-PATHSIZE)) & ((1U<<(CL-1))-1);
		}
		
		for(int s=0; s<SLIDESIZE; s+=PATHSIZE){
			int i = (endStage - SHFTR - s) % SHMEMWIDTH;
			path_t pp = pathPrev[i/PATHSIZE][winnerState];
			data[(dataEndInd-s)/PATHSIZE] = pp;
			winnerState = __brev(pp<<(32-PATHSIZE)) & ((1U<<(CL-1))-1);
		}
	}
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<ACS ACStype>
__global__ void viterbi_core(path_t* data, float* coded, path_t* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ float sharedMem[];
	float* sharedMemTip = sharedMem;
	sharedMemTip += ty * ((1<<(CL-1)) + (SHMEMWIDTH * 2));
	float* pathMetric = sharedMemTip;
	sharedMemTip += (1<<(CL-1));
	float (*branchMetric)[2] = (float(*)[2])sharedMemTip;

	path_t (*pathPrev) [1<<(CL-1)] = (path_t (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) *((SHMEMWIDTH-1)/PATHSIZE+1);
	
	//shift "data" and "coded" pointer to the index that this block should process
	int start_ind = DECSIZE * bdy * bx + DECSIZE * ty;
	data += start_ind/PATHSIZE;
	coded += start_ind*2;
	
	/****************************** calculate trellis parameters ******************************/	 
	trellis_t old, now;
	old.pm0 = 0.0; old.pm1 = 0.0;
	now = old;
	/******************************************************************************************/
	
	//initialize path metrics
	//each thread initializes one element
	pathMetric[tx] = 0.0;
	pathMetric[tx+(1<<(CL-2))] = 0.0;	
	
	bmCalc(0, SHFTL+SHFTR, branchMetric, coded);
	__syncthreads();
	for(int i=0; i<SHFTL+SHFTR; i++)
		forwardACS<ACStype>(i, &old, &now, pathPrev, branchMetric);

	for(int slide=0; slide<DECSIZE; slide+=SLIDESIZE){
		int ind = slide + SHFTL + SHFTR;
		bmCalc(ind, SLIDESIZE, branchMetric, coded);   
		__syncthreads();
		for(int i=ind; i<ind+SLIDESIZE; i++){	
			forwardACS<ACStype>(i, &old, &now, pathPrev, branchMetric);
		}
		traceback(ind+SLIDESIZE-1, slide+SLIDESIZE-1, data, pathMetric, pathPrev);
	}
}

//-----------------------------------------------------------------------------

int viterbi_run(float* input_d, path_t* output_d, int messageLen, float* time, ACS acsType) {
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
