#include "viterbi.h"
#include "viterbiConsts.h"

template<DecodeOut outputType>
__device__ void traceback(int endStage, int dataEndInd, int tbLength, decPack_t<outputType>* data, decPack_t<outputType> pathPrev[][1<<(CL-1)]){
	if(tx == 0){
		int winnerState = 0;
	
		for(int s=0; s<extraR-bpp<outputType>; s+=bpp<outputType>){
			decPack_t<outputType> pp = pathPrev[((endStage - s) % forwardLen)/bpp<outputType>][winnerState];
			winnerState = __brev(pp<<(32-bpp<outputType>)) & ((1U<<(CL-1))-1);
		}
		
		for(int s=0; s<tbLength; s+=bpp<outputType>){
			int i = (endStage - extraR - s) % forwardLen;
			decPack_t<outputType> pp = pathPrev[i/bpp<outputType>][winnerState];
			data[(dataEndInd-s)/bpp<outputType>] = pp;
			winnerState = __brev(pp<<(32-bpp<outputType>)) & ((1U<<(CL-1))-1);
		}
	}
}