#ifndef BITNICSORT_TOAPPLY_H
#define BITNICSORT_TOAPPLY_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>


typedef unsigned int uint;

extern "C" void ArbitraryBitonicSort_toApply(int NBox, int** IDStartEnd_ForBox_Host, int** IDStartEnd_ForBox_Dev, double* ToSortDev_ClustersPosX, int* SortedIndex, int dir);

extern "C" uint bitonicSort_toApply(
	double *d_DstKey,
	double *d_SrcKey,
	uint batchSize,
	uint arrayLength,
	uint dir);


//Enables maximum occupancy
#define SHARED_SIZE_LIMIT 1024

//Map to single instructions on G8x / G9x / G100
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )



__device__ inline void Comparator_toApply(
	double &keyA,
	double &keyB,
	uint dir
)
{
	double t;

	if ((keyA > keyB) == dir)
	{
		t = keyA;
		keyA = keyB;
		keyB = t;
	}
}



#endif
