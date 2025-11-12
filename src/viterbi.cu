#include "cuda_runtime.h"
#include "gputimer.h"
#include "gpuerrors.h"
#include "viterbi.h"
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


constexpr int CL = ViterbiCUDA<Metric::B16>::constLen;
constexpr int extraL = ViterbiCUDA<Metric::B16>::extraL;
constexpr int extraR = ViterbiCUDA<Metric::B16>::extraR;
constexpr int slideSize = ViterbiCUDA<Metric::B16>::slideSize;
constexpr int shmemWidth = ViterbiCUDA<Metric::B16>::shMemWidth;
template<Metric metricType> constexpr int bpp = ViterbiCUDA<metricType>::bitsPerPack;

template<Metric metricType> using metric_t = typename ViterbiCUDA<metricType>::metric_t;
template<Metric metricType> using decPack_t = typename ViterbiCUDA<metricType>::decPack_t;

template<Metric metricType>
ViterbiCUDA<metricType>::ViterbiCUDA()
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(false), blocksNum_total(100) 
{
	deviceSetup();
}

template<Metric metricType>
ViterbiCUDA<metricType>::ViterbiCUDA(size_t inputNum)
: pathPrev_d(nullptr), dec_d(nullptr), enc_d(nullptr), preAllocated(true)
{
	deviceSetup();
	memAlloc(inputNum);
}

template<Metric metricType>
ViterbiCUDA<metricType>::~ViterbiCUDA(){
	memFree();
}

template<Metric metricType>
void ViterbiCUDA<metricType>::memAlloc(size_t inputNum){
	size_t inputSize = getInputSize(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t ppSize = getPathPrevSize();
	
	//initialization
	HANDLE_ERROR(cudaMalloc((void**)&enc_d, inputSize));
	HANDLE_ERROR(cudaMalloc((void**)&dec_d, outputSize));
	HANDLE_ERROR(cudaMalloc((void**)&pathPrev_d, ppSize));
}

template<Metric metricType>
void ViterbiCUDA<metricType>::memFree(){
	if(dec_d) 		{cudaFree(dec_d); dec_d = nullptr;}
	if(enc_d) 		{cudaFree(enc_d); enc_d = nullptr;}
	if(pathPrev_d)	{cudaFree(pathPrev_d); pathPrev_d = nullptr;}
}

template<Metric metricType> size_t ViterbiCUDA<metricType>::getInputSize(size_t inputNum)
{return (inputNum * sizeof(float));}

template<Metric metricType> size_t ViterbiCUDA<metricType>::getMessageLen(size_t inputNum)
{return ((inputNum / 2 - (extraL + extraR)) / bitsPerPack * bitsPerPack);}

template<Metric metricType> size_t ViterbiCUDA<metricType>::getOutputSize(size_t inputNum)
{return (getMessageLen(inputNum)/8);}

template<Metric metricType> size_t ViterbiCUDA<metricType>::getSharedMemSize()
{return (shmemWidth * 4 * sizeof(metric_t) * blockDimY);}

template<Metric metricType> size_t ViterbiCUDA<metricType>::getPathPrevSize()
{return (shmemWidth / 8 * (1<<(constLen-1)) * blocksNum_total);}

template<Metric metricType>
void ViterbiCUDA<metricType>::deviceSetup(){
	cudaSetDevice(0);
	// cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100ULL * 1024 * 1024);
	// cudaDeviceProp deviceProp;
	// HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
	// printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
}

template<Metric metricType>
__device__ void bmCalc(int stage, int num, metric_t<metricType> branchMetric[][4], float* coded){
	for(int i=stage+tx; i<stage+num; i+=bdx){
		branchMetric[i%shmemWidth][0] = (metric_t<metricType>)((-coded[2*i] - coded[2*i+1]) * 20.0f);
		branchMetric[i%shmemWidth][1] = (metric_t<metricType>)((-coded[2*i] + coded[2*i+1]) * 20.0f);
		branchMetric[i%shmemWidth][2] = (metric_t<metricType>)(( coded[2*i] - coded[2*i+1]) * 20.0f);
		branchMetric[i%shmemWidth][3] = (metric_t<metricType>)(( coded[2*i] + coded[2*i+1]) * 20.0f);
	}   
}

template<Metric metricType>
__device__ void forwardACS(int stage, trellis<metricType>& old, trellis<metricType>& now, decPack_t<metricType> pathPrev[][1<<(CL-1)], metric_t<metricType> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1) {}

template<>
__device__ void forwardACS<Metric::B16>(int stage, trellis<Metric::B16>& old, trellis<Metric::B16>& now, decPack_t<Metric::B16> pathPrev[][1<<(CL-1)], metric_t<Metric::B16> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int ind = stage % shmemWidth;
	int stageSE = stage % (CL-1);

	if(stageSE < CL-6){
		unsigned int bms = (uint16_t)branchMetric[ind][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[ind][allBmInd0&3];

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
		int laneMask = (1<<(stageSE-CL+6));
		now.pm = __shfl_xor_sync(0xffffffff, now.pm, laneMask);
		now.pp = __shfl_xor_sync(0xffffffff, now.pp, laneMask);
		
		unsigned int bms = (uint16_t)branchMetric[ind][allBmInd0&3];
		bms = (bms << 16) | (uint16_t)branchMetric[ind][allBmInd1&3];
		
		unsigned int pmMax = __viaddmax_s16x2(old.pm, __vadd2(bms, bms), now.pm);
		unsigned int cond = __vcmpeq2(pmMax, now.pm);
		now.pm = __vsub2(pmMax, bms);
		unsigned int permMap = ((cond>>8) & 0x0000cccc) | 0x00003210;
		now.pp = __byte_perm(old.pp, now.pp, permMap);
		cond &= 0x00010001;
		cond ^= (tx & laneMask) ? 0x00010001 : 0;
		now.pp = (now.pp << 1) | cond;

		old = now;
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((ind+1) % 16 == 0){
		unsigned int padd = tx % (1<<stageSE);
		padd <<= (CL-1);
		unsigned int state0 = padd + tx;
		unsigned int state1 = state0 + (1U<<(CL-2));
		state0 >>= stageSE;
		state1 >>= stageSE;
		pathPrev[ind/16][state0] = now.pp >> 16;
		pathPrev[ind/16][state1] = now.pp  & 0x0000ffff;
		old.pp = 0;
		now.pp = 0;
	}
}

template<>
__device__ void forwardACS<Metric::B32>(int stage, trellis<Metric::B32>& old, trellis<Metric::B32>& now, decPack_t<Metric::B32> pathPrev[][1<<(CL-1)], metric_t<Metric::B32> branchMetric[][4], unsigned int& allBmInd0, unsigned int& allBmInd1){
	int ind = stage % shmemWidth;
	int stageSE = stage % (CL-1);

	if(stageSE < CL-6){
		int bm = branchMetric[ind][allBmInd0&3];

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
		int laneMask = (1<<(stageSE-CL+6));
		now.pm0 = __shfl_xor_sync(0xffffffff, now.pm0, laneMask);
		now.pm1 = __shfl_xor_sync(0xffffffff, now.pm1, laneMask);
		now.pp0 = __shfl_xor_sync(0xffffffff, now.pp0, laneMask);
		now.pp1 = __shfl_xor_sync(0xffffffff, now.pp1, laneMask);
		
		int bm0 = branchMetric[ind][allBmInd0&3];
		int bm1 = branchMetric[ind][allBmInd1&3];
		
		int pmMax0 = __viaddmax_s32(old.pm0, bm0*2, now.pm0);
		bool cond0 = (pmMax0 == now.pm0);
		now.pm0 = pmMax0 - bm0;
		now.pp0 = cond0 ? now.pp0 : old.pp0;
		cond0 ^= ((tx & laneMask) != 0);
		now.pp0 = (now.pp0 << 1) | cond0;

		int pmMax1 = __viaddmax_s32(old.pm1, bm1*2, now.pm1);
		bool cond1 = (pmMax1 == now.pm1);
		now.pm1 = pmMax1 - bm1;
		now.pp1 = cond1 ? now.pp1 : old.pp1;
		cond1 ^= ((tx & laneMask) != 0);
		now.pp1 = (now.pp1 << 1) | cond1;

		old = now;
	}

	allBmInd0 = (allBmInd0 >> 2) | ((allBmInd0&3) << (2*(CL-2)));
	allBmInd1 = (allBmInd1 >> 2) | ((allBmInd1&3) << (2*(CL-2)));

	if((ind+1) % 32 == 0){
		unsigned int padd = tx % (1<<stageSE);
		padd <<= (CL-1);
		unsigned int state0 = padd + tx;
		unsigned int state1 = state0 + (1U<<(CL-2));
		state0 >>= stageSE;
		state1 >>= stageSE;
		pathPrev[ind/32][state0] = now.pp0;
		pathPrev[ind/32][state1] = now.pp1;
		old.pp0 = 0;
		old.pp1 = 0;
		now.pp0 = 0;
		now.pp1 = 0;
	}
}

__device__ void bmIndCalc(unsigned int& allBmInd0, unsigned int& allBmInd1){
	allBmInd0 = 0;
	allBmInd1 = 0;
	for(int ind=CL-2; ind>=0; ind--){
		unsigned int inState0 = ((tx<<(2*CL-7)) + (tx<<(CL-6))) >> ind;
		inState0 &= (1<<CL)-1;
		unsigned int inState1 = inState0 ^ (1<<(CL-1-ind));

		bool out0 = __popc(inState0 & ViterbiCUDA<Metric::B16>::polyn1) % 2;
		bool out1 = __popc(inState0 & ViterbiCUDA<Metric::B16>::polyn2) % 2;
		int bmInd = (out0 << 1) | out1;
		allBmInd0 = (allBmInd0 << 2) | bmInd;

		out0 = __popc(inState1 & ViterbiCUDA<Metric::B16>::polyn1) % 2;
		out1 = __popc(inState1 & ViterbiCUDA<Metric::B16>::polyn2) % 2;
		bmInd = (out0 << 1) | out1;
		allBmInd1 = (allBmInd1 << 2) | bmInd;
	}
}

template<Metric metricType>
__device__ void traceback(int endStage, int dataEndInd, int tbLength, decPack_t<metricType>* data, decPack_t<metricType> pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		int winnerState = 0;
	
		for(int s=0; s<extraR-bpp<metricType>; s+=bpp<metricType>){
			decPack_t<metricType> pp = pathPrev[((endStage - s) % shmemWidth)/bpp<metricType>][winnerState];
			winnerState = __brev(pp<<(32-bpp<metricType>)) & ((1U<<(CL-1))-1);
		}
		
		for(int s=0; s<tbLength; s+=bpp<metricType>){
			int i = (endStage - extraR - s) % shmemWidth;
			decPack_t<metricType> pp = pathPrev[i/bpp<metricType>][winnerState];
			data[(dataEndInd-s)/bpp<metricType>] = pp;
			winnerState = __brev(pp<<(32-bpp<metricType>)) & ((1U<<(CL-1))-1);
		}
	}
}

//-----------------------------------------------------------------------------
//the main core of viterbi decoder
//get data and polynoials ans decode 
template<Metric metricType>
__global__ void viterbi_core(decPack_t<metricType>* data, float* coded, size_t messageLen, decPack_t<metricType>* pathPrev_all) {
	//coded: input coded array that contains 2*n bits with constraint mentioned above
	//data: output array that contains n allocated bits with constraint mentioned above
	
	extern __shared__ int sharedMem[];
	metric_t<metricType>* sharedMemTip = reinterpret_cast<metric_t<metricType>*>(sharedMem);
	sharedMemTip += ty * (shmemWidth * 4);
	metric_t<metricType> (*branchMetric)[4] = (metric_t<metricType>(*)[4])sharedMemTip;

	decPack_t<metricType> (*pathPrev) [1<<(CL-1)] = (decPack_t<metricType> (*) [1<<(CL-1)])pathPrev_all + (bx*bdy + ty) * ((shmemWidth-1)/bpp<metricType>+1);
	
	size_t packNum = messageLen / bpp<metricType>;
	size_t decLen = packNum / (gdx * bdy);
	size_t remPacks = packNum % (gdx * bdy);
	size_t startInd = decLen * (bx*bdy + ty) + min(remPacks, size_t(bx*bdy + ty));
	if( (bx*bdy + ty) < remPacks ) decLen++;
	decLen *= bpp<metricType>;
	startInd *= bpp<metricType>;
	
	data += startInd/bpp<metricType>;
	coded += startInd*2;
	
	/****************************** calculate trellis parameters ******************************/	 
	trellis<metricType> old, now;
	unsigned int allBmInd0, allBmInd1;
	bmIndCalc(allBmInd0, allBmInd1);
	/******************************************************************************************/

	bmCalc<metricType>(0, extraL+extraR, branchMetric, coded);
	__syncwarp();
	for(int stage=0; stage<extraL+extraR; stage++)
		forwardACS<metricType>(stage, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1);

	int slide;
	for(slide=0; slide<decLen; slide+=slideSize){
		int stage = slide + extraL + extraR;
		bmCalc<metricType>(stage, slideSize, branchMetric, coded);   
		__syncwarp();
		for(int i=stage; i<stage+slideSize; i++){	
			forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1);
		}
		traceback<metricType>(stage+slideSize-1, slide+slideSize-1, slideSize, data, pathPrev);
	}

	int stage = slide + extraL + extraR;
	int remSlideSize = decLen % slideSize;
	bmCalc<metricType>(stage, remSlideSize, branchMetric, coded);   
	__syncwarp();
	for(int i=stage; i<stage+remSlideSize; i++){	
		forwardACS<metricType>(i, old, now, pathPrev, branchMetric, allBmInd0, allBmInd1);
	}
	traceback<metricType>(stage+remSlideSize-1, slide+remSlideSize-1, remSlideSize, data, pathPrev);
}

//-----------------------------------------------------------------------------
template<Metric metricType>
void ViterbiCUDA<metricType>::run(float* input_h, decPack_t* output_h, size_t inputNum, float* kernelTime){
	size_t inputSize = getInputSize(inputNum);
	size_t messageLen = getMessageLen(inputNum);
	size_t outputSize = getOutputSize(inputNum);
	size_t sharedMemSize = getSharedMemSize();

	if(!preAllocated) memAlloc(inputNum);

	HANDLE_ERROR(cudaMemcpy(enc_d, input_h, inputSize, cudaMemcpyHostToDevice));
	
	GpuTimer timer;
	dim3 grid (blocksNum_total/blockDimY, 1, 1); 
	dim3 block (32, blockDimY, 1);

	timer.Start();
	viterbi_core<metricType> <<<grid, block, sharedMemSize>>> (dec_d, enc_d, messageLen, pathPrev_d);
	timer.Stop();
	if(kernelTime) *kernelTime = timer.Elapsed();
	HANDLE_ERROR(   cudaPeekAtLastError()   );

	HANDLE_ERROR(cudaMemcpy(output_h, dec_d, outputSize, cudaMemcpyDeviceToHost));

	if(!preAllocated) memFree();
}

template class ViterbiCUDA<Metric::B16>;
template class ViterbiCUDA<Metric::B32>;