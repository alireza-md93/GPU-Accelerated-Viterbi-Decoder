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

template<Metric metricType>
struct trellis;

//two packed 16-bit values
template<>
struct trellis<Metric::B16>{
	unsigned int pm;
	unsigned int pp;

	__device__ trellis(): pm(0), pp(0) {}
};

template<>
struct trellis<Metric::B32>{
	int pm0;
	int pm1;
	unsigned int pp0;
	unsigned int pp1;

	__device__ trellis(): pm0(0), pm1(0), pp0(0), pp1(0) {}
};

template<Metric metricType>
__device__ __forceinline__ void bmCalc(int ind, int num, metric<metricType> branchMetric[][4], float* coded){
	for(int i=ind+tx; i<ind+num; i+=bdx){
		branchMetric[i%SHMEMWIDTH][0] = (metric<metricType>)((-coded[2*i] - coded[2*i+1]) * 20.0f);
		branchMetric[i%SHMEMWIDTH][1] = (metric<metricType>)((-coded[2*i] + coded[2*i+1]) * 20.0f);
		branchMetric[i%SHMEMWIDTH][2] = (metric<metricType>)(( coded[2*i] - coded[2*i+1]) * 20.0f);
		branchMetric[i%SHMEMWIDTH][3] = (metric<metricType>)(( coded[2*i] + coded[2*i+1]) * 20.0f);
	}   
}

template<Metric metricType>
__device__ __forceinline__ void forwardACS(int ind, trellis<metricType>& old, trellis<metricType>& now, pack_t<metricType> pathPrev[][1<<(CL-1)], metric<metricType> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){}

template<>
__device__ __forceinline__ void forwardACS<Metric::B16>(int ind, trellis<Metric::B16>& old, trellis<Metric::B16>& now, pack_t<Metric::B16> pathPrev[][1<<(CL-1)], metric<Metric::B16> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int i = ind % SHMEMWIDTH;

	//input bit plus state
	unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> (ind%(CL-1));
	inState0 &= (1<<CL)-1;
	unsigned int inState1 = inState0 ^ (1<<(CL-1-ind%(CL-1)));

	if(i%(CL-1) < CL-6){
		unsigned int bms = (uint16_t)branchMetric[i][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[i][allBmInd0&3];

		unsigned int pmRev = __funnelshift_l(old.pm, old.pm, 16);
		now.pm = __viaddmax_s16x2(old.pm, __vadd2(bms, bms), pmRev);
		unsigned int cond = __vcmpeq2(now.pm, pmRev);
		now.pm = __vsub2(now.pm, bms);
		unsigned int permMap = (((cond>>8) & 0x00002222) ^ 0x00002200) | 0x00001010;
		now.pp = __byte_perm(old.pp, 0, permMap);
		cond &= 0x00010001;
		cond ^= 1;
		now.pp = (now.pp << 1) | cond;

		old = now;
	}
	else{
		now.pm = __shfl_xor_sync(0xffffffff, now.pm, 1<<(ind%(CL-1)-CL+6));
		now.pp = __shfl_xor_sync(0xffffffff, now.pp, 1<<(ind%(CL-1)-CL+6));
		
		unsigned int bms = (uint16_t)branchMetric[i][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[i][allBmInd1&3];
		
		unsigned int pmMax = __viaddmax_s16x2(old.pm, __vadd2(bms, bms), now.pm);
		unsigned int cond = __vcmpeq2(pmMax, now.pm);
		now.pm = __vsub2(pmMax, bms);
		unsigned int permMap = ((cond>>8) & 0x0000cccc) | 0x00003210;
		now.pp = __byte_perm(old.pp, now.pp, permMap);
		cond &= 0x00010001;
		cond ^= (inState0 & 1) ? 0x00010001 : 0;
		now.pp = (now.pp << 1) | cond;

		old = now;
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((i+1) % 16 == 0){
		pathPrev[i/16][inState0>>1] = now.pp >> 16;
		pathPrev[i/16][inState1>>1] = now.pp  & 0x0000ffff;
		old.pp = 0;
		now.pp = 0;
	}
}

template<>
__device__ __forceinline__ void forwardACS<Metric::B32>(int ind, trellis<Metric::B32>& old, trellis<Metric::B32>& now, pack_t<Metric::B32> pathPrev[][1<<(CL-1)], metric<Metric::B32> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int i = ind % SHMEMWIDTH;

	//input bit plus state
	unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> (ind%(CL-1));
	inState0 &= (1<<CL)-1;
	unsigned int inState1 = inState0 ^ (1<<(CL-1-ind%(CL-1)));

	if(i%(CL-1) < CL-6){
		int bm = branchMetric[i][allBmInd0&3];

		now.pm0 = __viaddmax_s32(old.pm0, bm*2, old.pm1);
		int cond0 = (now.pm0 == old.pm1 ? 1 : 0);
		now.pm0 = now.pm0 - bm;
		now.pp0 = cond0 ? old.pp1 : old.pp0;
		now.pp0 = (now.pp0 << 1) | cond0;


		now.pm1 = __viaddmax_s32(old.pm1, bm*2, old.pm0);
		int cond1 = (now.pm1 == old.pm0 ? 0 : 1);
		now.pm1 = now.pm1 - bm;
		now.pp1 = cond1 ? old.pp1 : old.pp0;
		now.pp1 = (now.pp1 << 1) | cond1;

		old = now;
	}
	else{
		now.pm0 = __shfl_xor_sync(0xffffffff, now.pm0, 1<<(ind%(CL-1)-CL+6));
		now.pm1 = __shfl_xor_sync(0xffffffff, now.pm1, 1<<(ind%(CL-1)-CL+6));
		now.pp0 = __shfl_xor_sync(0xffffffff, now.pp0, 1<<(ind%(CL-1)-CL+6));
		now.pp1 = __shfl_xor_sync(0xffffffff, now.pp1, 1<<(ind%(CL-1)-CL+6));
		
		int bm0 = branchMetric[i][allBmInd0&3];
		int bm1 = branchMetric[i][allBmInd1&3];
		
		int pmMax0 = __viaddmax_s32(old.pm0, bm0*2, now.pm0);
		unsigned int cond0 = (pmMax0 == now.pm0 ? 1 : 0);
		now.pm0 = pmMax0 - bm0;
		now.pp0 = cond0 ? now.pp0 : old.pp0;
		cond0 ^= (inState0 & 1);
		now.pp0 = (now.pp0 << 1) | cond0;

		int pmMax1 = __viaddmax_s32(old.pm1, bm1*2, now.pm1);
		unsigned int cond1 = (pmMax1 == now.pm1 ? 1 : 0);
		now.pm1 = pmMax1 - bm1;
		now.pp1 = cond1 ? now.pp1 : old.pp1;
		cond1 ^= (inState1 & 1);
		now.pp1 = (now.pp1 << 1) | cond1;

		old = now;
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((i+1) % 32 == 0){
		pathPrev[i/32][inState0>>1] = now.pp0;
		pathPrev[i/32][inState1>>1] = now.pp1;
		old.pp0 = 0;
		old.pp1 = 0;
		now.pp0 = 0;
		now.pp1 = 0;
	}
}

__device__ __forceinline__ void bmIndCalc(unsigned int& allBmInd0, unsigned int& allBmInd1){
	allBmInd0 = 0;
	allBmInd1 = 0;
	for(int ind=CL-2; ind>=0; ind--){
		unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> ind;
		inState0 &= (1<<CL)-1;
		unsigned int inState1 = inState0 ^ (1<<(CL-1-ind));

		bool out0 = __popc(inState0 & POLYN1) % 2;
		bool out1 = __popc(inState0 & POLYN2) % 2;
		int bmInd = (out0 << 1) | out1;
		allBmInd0 = (allBmInd0 << 2) | bmInd;

		out0 = __popc(inState1 & POLYN1) % 2;
		out1 = __popc(inState1 & POLYN2) % 2;
		bmInd = (out0 << 1) | out1;
		allBmInd1 = (allBmInd1 << 2) | bmInd;
	}
}

template<Metric metricType>
__device__ __forceinline__ void traceback(int endStage, int dataEndInd, pack_t<metricType>* data, pack_t<metricType> pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		int winnerState = 0;
	
		for(int s=0; s<SHFTR-metricType; s+=metricType){
			pack_t<metricType> pp = pathPrev[((endStage - s) % SHMEMWIDTH)/metricType][winnerState];
			winnerState = __brev(pp<<(32-metricType)) & ((1U<<(CL-1))-1);
		}
		
		for(int s=0; s<SLIDESIZE; s+=metricType){
			int i = (endStage - SHFTR - s) % SHMEMWIDTH;
			pack_t<metricType> pp = pathPrev[i/metricType][winnerState];
			data[(dataEndInd-s)/metricType] = pp;
			winnerState = __brev(pp<<(32-metricType)) & ((1U<<(CL-1))-1);
		}
	}
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<Metric metricType>
__global__ void viterbi_core(pack_t<metricType>* data, float* coded, pack_t<metricType>* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ int sharedMem[];
	metric<metricType>* sharedMemTip = reinterpret_cast<metric<metricType>*>(sharedMem);
	sharedMemTip += ty * (SHMEMWIDTH * 4);
	metric<metricType> (*branchMetric)[4] = (metric<metricType>(*)[4])sharedMemTip;

	pack_t<metricType> (*pathPrev) [1<<(CL-1)] = (pack_t<metricType> (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) * ((SHMEMWIDTH-1)/metricType+1);
	
	//shift "data" and "coded" pointer to the index that this block should process
	int start_ind = DECSIZE * bdy * bx + DECSIZE * ty;
	data += start_ind/metricType;
	coded += start_ind*2;
	
	/****************************** calculate trellis parameters ******************************/	 
	trellis<metricType> old, now;
	unsigned int allBmInd0, allBmInd1;
	bmIndCalc(allBmInd0, allBmInd1);
	/******************************************************************************************/

	bmCalc<metricType>(0, SHFTL+SHFTR, branchMetric, coded);
	__syncwarp();
	for(int i=0; i<SHFTL+SHFTR; i++)
		forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1);

	for(int slide=0; slide<DECSIZE; slide+=SLIDESIZE){
		int ind = slide + SHFTL + SHFTR;
		bmCalc<metricType>(ind, SLIDESIZE, branchMetric, coded);   
		__syncwarp();
		for(int i=ind; i<ind+SLIDESIZE; i++){	
			forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1);
		}
		traceback<metricType>(ind+SLIDESIZE-1, slide+SLIDESIZE-1, data, pathPrev);
	}
}

//-----------------------------------------------------------------------------
template<Metric metricType>
void viterbi_run(float* input_d, pack_t<metricType>* output_d, int messageLen, float* time) {
	int wins = messageLen / DECSIZE;
	pack_t<metricType>* pathPrev_d;
	int ppSize = SHMEMWIDTH / 8 * (1<<(CL-1)) * wins;
	int sharedMemSize = SHMEMWIDTH * 4 * sizeof(metric<metricType>);
	sharedMemSize *= BLOCK_DIMY;

	HANDLE_ERROR(cudaMalloc((void**)&pathPrev_d, ppSize));

	GpuTimer timer;
	
	dim3 grid (wins/BLOCK_DIMY, 1, 1); 
	dim3 block (32, BLOCK_DIMY, 1);

	timer.Start();
	viterbi_core<metricType> <<<grid, block, sharedMemSize>>> ((pack_t<metricType>*)output_d, input_d, pathPrev_d);
	timer.Stop();
	*time = timer.Elapsed();
	
	HANDLE_ERROR(   cudaPeekAtLastError()   );
	HANDLE_ERROR(cudaFree(pathPrev_d));
}

template void viterbi_run<Metric::B16>(float* input_d, pack_t<Metric::B16>* output_d, int messageLen, float* time);
template void viterbi_run<Metric::B32>(float* input_d, pack_t<Metric::B32>* output_d, int messageLen, float* time);