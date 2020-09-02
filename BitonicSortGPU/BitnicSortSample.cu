

#include <assert.h>
#include <cooperative_groups.h>
#include "bitnicSort.h"


////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void bitonicSortShared(
	double *d_DstKey,
	double *d_SrcKey,
	uint arrayLength,
	uint dir
)
{
	// Handle to thread block group
	//Shared memory storage for one or more short vectors
	__shared__ double s_key[SHARED_SIZE_LIMIT];

	//Offset to the beginning of subbatch and load data
	d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x + 0] = d_SrcKey[0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

	for (uint size = 2; size < arrayLength; size <<= 1)
	{
		//Bitonic merge
		uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

		for (uint stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0],
				s_key[pos + stride],
				ddd
			);
		}
	}

	//ddd == dir for the last bitonic merge step
	{
		for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0],
				s_key[pos + stride],
				dir
			);
		}
	}

	__syncthreads();
	d_DstKey[0] = s_key[threadIdx.x + 0];
	d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(
	double *d_DstKey,
	double *d_SrcKey
)
{
	// Handle to thread block group
	//Shared memory storage for current subarray
	__shared__ double s_key[SHARED_SIZE_LIMIT];

	//Offset to the beginning of subarray and load data
	d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x + 0] = d_SrcKey[0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

	for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
	{
		//Bitonic merge
		uint ddd = (threadIdx.x & (size / 2)) != 0;

		for (uint stride = size / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0],
				s_key[pos + stride],
				ddd
			);
		}
	}

	//Odd / even arrays of SHARED_SIZE_LIMIT elements
	//sorted in opposite directions
	uint ddd = blockIdx.x & 1;
	{
		for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(
				s_key[pos + 0],
				s_key[pos + stride],
				ddd
			);
		}
	}


	__syncthreads();
	d_DstKey[0] = s_key[threadIdx.x + 0];
	d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
	double *d_DstKey,
	double *d_SrcKey,
	uint arrayLength,
	uint size,
	uint stride,
	uint dir
)
{
	uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
	uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

	//Bitonic merge
	uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
	uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

	double keyA = d_SrcKey[pos + 0];
	double keyB = d_SrcKey[pos + stride];

	Comparator(
		keyA,
		keyB,
		ddd
	);

	d_DstKey[pos + 0] = keyA;
	d_DstKey[pos + stride] = keyB;
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
	double *d_DstKey,
	double *d_SrcKey,
	uint arrayLength,
	uint size,
	uint dir
)
{
	//Shared memory storage for current subarray
	__shared__ double s_key[SHARED_SIZE_LIMIT];

	d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x + 0] = d_SrcKey[0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];

	//Bitonic merge
	uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
	uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

	for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
	{
		__syncthreads();
		uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
		Comparator(
			s_key[pos + 0],
			s_key[pos + stride],
			ddd
		);
	}

	__syncthreads();
	d_DstKey[0] = s_key[threadIdx.x + 0];
	d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
	if (!L)
	{
		*log2L = 0;
		return 0;
	}
	else
	{
		for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

		return L;
	}
}

extern "C" uint bitonicSort(
	double *d_DstKey,
	double *d_SrcKey,
	uint batchSize,
	uint arrayLength,
	uint dir
)
{
	//Nothing to sort
	if (arrayLength < 2)
		return 0;

	//Only power-of-two array lengths are supported by this implementation
	uint log2L;
	uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
	assert(factorizationRemainder == 1);

	dir = (dir != 0);

	uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
	uint threadCount = SHARED_SIZE_LIMIT / 2;

	if (arrayLength <= SHARED_SIZE_LIMIT)
	{
		assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
		bitonicSortShared << <blockCount, threadCount >> > (d_DstKey, d_SrcKey, arrayLength, dir);
	}
	else
	{
		bitonicSortShared1 << <blockCount, threadCount >> > (d_DstKey, d_SrcKey);

		for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
			for (unsigned stride = size / 2; stride > 0; stride >>= 1)
				if (stride >= SHARED_SIZE_LIMIT)
				{
					bitonicMergeGlobal << <(batchSize * arrayLength) / 512, 256 >> > (d_DstKey, d_DstKey, arrayLength, size, stride, dir);
				}
				else
				{
					bitonicMergeShared << <blockCount, threadCount >> > (d_DstKey, d_DstKey, arrayLength, size, dir);
					break;
				}
	}

	return threadCount;
}
