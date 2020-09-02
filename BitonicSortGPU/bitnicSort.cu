#include "bitnicSort.h"
#include <iostream>

#define BLOCKSIZE 1024

__device__ inline void Comparetor(double &ValA, double &ValB, int dir) {
	double temp;

	// dir 0 for decreasing , dir 1 for increasing

	if ((ValA > ValB) == dir) {
		temp = ValA;
		ValA = ValB;
		ValB = temp;
	}
}

inline void Comparetor_Host(double &ValA, double &ValB, int dir) {
	double temp;

	// dir 0 for decreasing , dir 1 for increasing

	if ((ValA > ValB) == dir) {
		temp = ValA;
		ValA = ValB;
		ValB = temp;
	}
}


/*Used For Array Size less than BLOCKSIZE And is of power 2*/
__global__ void Kernel_PowerTowShared(int NSize,double* Dev_TestArray,int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = 0.E0;
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[cid + BLOCKSIZE / 2];
	}

	for (int i = 2; i < NSize; i <<= 1) {

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2*cid - (cid & (stride - 1));

			if ((pos + stride) < NSize) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}
	
	for (int stride = NSize; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * cid - (cid & (stride - 1));

		if ((pos + stride) < NSize) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		}

	}
	
	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


__global__ void Kernel_ArbitraryBitonicSort(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = 0.E0;
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[cid + BLOCKSIZE / 2];
	}

	int NThreadUpHalf = NSize / 2;

	int IDRelative = cid%NThreadUpHalf;

	int IDStart = NThreadUpHalf*(cid/ NThreadUpHalf);

	int IDEnd = NThreadUpHalf + (cid / NThreadUpHalf)*(NSize - NThreadUpHalf);

	int FirstGreater = (NThreadUpHalf - 1) << 1;

	//for (int i = 2; i < NThreadUpHalf; i <<= 1) {
	for (int i = 2; i <=16; i <<= 1) {
	//for (int i = 2; i <= FirstGreater; i <<= 1) {

		tempDir = (cid / NThreadUpHalf)^(dir ^ ((IDRelative & (i / 2)) != 0));

		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + NThreadUpHalf - i/2;
		//}

		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + IDEnd - IDStart - i / 2;
		//}

		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			if (i > NThreadUpHalf && stride<i/2) {
				IDStart = IDStart + IDEnd - IDStart - i / 2;
			}

			//if (i > NThreadUpHalf) {
			//	IDStart = IDStart + IDEnd - IDStart - i / 2;
			//}

			pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

			if ((pos + stride) < IDEnd) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}
	
	/*
	for (int stride = 9; stride > 0; stride = stride/2) {
		__syncthreads();

		//pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

		if ((cid + stride) < NSize) {
			Comparetor(Share_TestArray[cid], Share_TestArray[cid + stride], dir);
		}

	}
	*/


	
	/*
	for (int stride = NThreadUpHalf; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

		if ((pos + stride) < IDEnd) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		}

	}
	*/
	
	
	

	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


__global__ void Kernel_ArbitraryBitonicSort2(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = 0.E0;
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[cid + BLOCKSIZE / 2];
	}

	for (int i = 2; i < NSize; i <<= 1) {

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * cid - (cid & (stride - 1));

			if ((pos + stride) < NSize) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}

	/*
	int IDStart = 0;

	//for (int i = 2; i < NThreadUpHalf; i <<= 1) {
	for (int i = 32; i <=32; i <<= 1) {
		//for (int i = 2; i <= FirstGreater; i <<= 1) {

		tempDir = (dir ^ ((cid & (i / 2)) != 0));

		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + NThreadUpHalf - i/2;
		//}

		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + IDEnd - IDStart - i / 2;
		//}

		for (int stride = i / 2; stride >= i/2; stride >>= 1) {

			__syncthreads();

			if (stride < i / 2) {
				IDStart = NSize - i/2 - 1;
			}

			//if (i > NThreadUpHalf) {
			//	IDStart = IDStart + IDEnd - IDStart - i / 2;
			//}

			pos = IDStart + 2 * cid - (cid & (stride - 1));

			if ((pos + stride) < NSize) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}
	


	/*
	for (int stride = NSize; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * cid - (cid & (stride - 1));

		if ((pos + stride) < NSize) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		}

	}*/

	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


__device__ int FirstGreaterPower2(int NSize) {
	int result;

	result = 1;

	while (result > 0 && result < NSize) {
		result <<= 1;
	}

	return result;
}

__device__ int GetIDStart(int NSize,int Stride,int cid) {
	int result;

	int NHalf;
	int TopHalfThread = FirstGreaterPower2(NSize/2);

	NHalf = NSize / 2;
	if (cid >= TopHalfThread) {
		NHalf = NSize - NHalf;
	}

	result = cid;

	return result;
}

__global__ void Kernel_ArbitraryBitonicSort3(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = 0.E0;
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[cid + BLOCKSIZE / 2];
	}

	int NThreadUpHalf = NSize / 2;

	int IDRelative = cid % NThreadUpHalf;

	int IDStart = NThreadUpHalf * (cid / NThreadUpHalf);

	int IDEnd = NThreadUpHalf + (cid / NThreadUpHalf)*(NSize - NThreadUpHalf);

	int FirstGreater = (NThreadUpHalf - 1) << 1;

	//for (int i = 2; i < NThreadUpHalf; i <<= 1) {
	for (int i = 2; i <= 8; i <<= 1) {
		//for (int i = 2; i <= FirstGreater; i <<= 1) {

		tempDir = (cid / NThreadUpHalf) ^ (dir ^ ((IDRelative & (i / 2)) != 0));






		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + NThreadUpHalf - i/2;
		//}

		//if (i > NThreadUpHalf) {
		//	IDStart = IDStart + IDEnd - IDStart - i / 2;
		//}

		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			if (i > NThreadUpHalf && stride < i / 2) {
				IDStart = IDStart + IDEnd - IDStart - i / 2;
			}

			//if (i > NThreadUpHalf) {
			//	IDStart = IDStart + IDEnd - IDStart - i / 2;
			//}

			pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

			if ((pos + stride) < IDEnd) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}

	/*
	for (int stride = 9; stride > 0; stride = stride/2) {
		__syncthreads();

		//pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

		if ((cid + stride) < NSize) {
			Comparetor(Share_TestArray[cid], Share_TestArray[cid + stride], dir);
		}

	}
	*/



	/*
	for (int stride = NThreadUpHalf; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = IDStart + 2 * IDRelative - (IDRelative & (stride - 1));

		if ((pos + stride) < IDEnd) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		}

	}
	*/




	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_ArbitraryBitonicSort(int NSize, double* Dev_TestArray, int dir,double padNum) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int tempDir;
	int pos;
	int LastPowerTwo;
	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = padNum;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = padNum;
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[cid + BLOCKSIZE / 2];
	}

	LastPowerTwo = 1;

	for (int i = 2; i < NSize; i <<= 1) {

		LastPowerTwo = i;

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * cid - (cid & (stride - 1));

			//if ((pos + stride) < NSize) {
				Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			//}

		}
	}

	for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * cid - (cid & (stride - 1));

		//if ((pos + stride) < NSize) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		//}

	}

	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


__global__ void Kernel_GlobalMerge_Pre(int Size, int SegmentsStride, int MaxSegments, int** IDStartEnd, double* Dev_TestArray, int dir, int *OEFlags) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int tempDir;
	int IDSeg = cid;
	int IDLevel = IDSeg / SegmentsStride;
	int IDSegStart = IDLevel * 2 * SegmentsStride;
	int IDSegEnd = (IDLevel + 1) * 2 * SegmentsStride - 1;
	int IDSegMap = IDSegStart + IDSeg % SegmentsStride;
	int ICLevelStart;
	int ICLevelRightHalfStart;
	int ICLevelRightHalfEnd;
	int ICLevelEnd;
	int FlagsShift;

	tempDir = dir ^ ((IDSegStart / Size) & 1);

	//ICLevelStart = IDStartEnd[IDSegStart][0];
	//ICLevelEnd = IDStartEnd[IDSegEnd][1];
	ICLevelStart = *(*(IDStartEnd + IDSegStart) + 0);
	ICLevelEnd = *(*(IDStartEnd + IDSegEnd) + 1);

	//ICLevelRightHalfStart = IDStartEnd[IDSegStart + SegmentsStride][0];
	ICLevelRightHalfStart = *(*(IDStartEnd + IDSegStart + SegmentsStride) + 0);
	ICLevelRightHalfEnd = ICLevelEnd;

	//int SegmentsStrideLast = SegmentsStride >> (SegmentsStride == (Size / 2));
	//int IDSegStartLast = 2*(IDSegMap / (2*SegmentsStrideLast))*SegmentsStrideLast;
	//int MainDir = dir^(  (IDSegStartLast / (Size / (1 + (SegmentsStride == (Size/2)) )   )) & 1);

	//int SizeChangeFlag = (SegmentsStride == (Size / 2));
	//int LastStridePeriodicFlag = ((IDSegMap /(SegmentsStride<<1)) & 1) ^ ((IDSegMap /Size) & 1);
	//int LastDir = (dir ^ ((IDSegMap /(Size>> SizeChangeFlag)) & 1) )^ (LastStridePeriodicFlag & (!SizeChangeFlag));
	FlagsShift = tempDir ^ (Dev_TestArray[ICLevelRightHalfEnd] >= Dev_TestArray[ICLevelRightHalfStart]);

	OEFlags[IDSegMap] = ((ICLevelEnd - ICLevelStart + 1) % 2 != 0)&FlagsShift;
}


__global__ void Kernel_GlobalMerge(int Size,int SegmentsStride,int MaxSegments, int** IDStartEnd, double* Dev_TestArray, int dir,int *OEFlags) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int tempDir;
	int pos;
	double Left;
	double Right;
	int IDSeg = cid / BLOCKSIZE;
	int IDLevel = IDSeg/ SegmentsStride;
	int IDSegStart = IDLevel * 2*SegmentsStride;
	int IDSegEnd = (IDLevel + 1) * 2*SegmentsStride - 1;
	int IDSegMap = IDSegStart + IDSeg% SegmentsStride;
	int ICSegStart;
	int ICSegEnd;
	int ICLevelStart;
	int ICLevelRightHalfStart;
	int ICLevelRightHalfEnd;
	int ICLevelEnd;
	int Stride;

	tempDir = dir ^ ((IDSegStart /Size) & 1);

	//ICSegStart = IDStartEnd[IDSegMap][0];
	//ICSegEnd = IDStartEnd[IDSegMap][1];
	ICSegStart = *(*(IDStartEnd + IDSegMap) + 0);
	ICSegEnd = *(*(IDStartEnd + IDSegMap) + 1);

	pos = ICSegStart + tid;

	//ICLevelStart = IDStartEnd[IDSegStart][0];
	//ICLevelEnd = IDStartEnd[IDSegEnd][1];
	ICLevelStart = *(*(IDStartEnd + IDSegStart) + 0);
	ICLevelEnd = *(*(IDStartEnd + IDSegEnd) + 1);

	//ICLevelRightHalfStart = IDStartEnd[IDSegStart + SegmentsStride][0];
	/*
	ICLevelRightHalfStart = *(*(IDStartEnd + IDSegStart + SegmentsStride) + 0);
	ICLevelRightHalfEnd = ICLevelEnd;
	*/

	//int SegmentsStrideLast = SegmentsStride >> (SegmentsStride == (Size / 2));
	//int IDSegStartLast = 2*(IDSegMap / (2*SegmentsStrideLast))*SegmentsStrideLast;
	//int MainDir = dir^(  (IDSegStartLast / (Size / (1 + (SegmentsStride == (Size/2)) )   )) & 1);

	//int SizeChangeFlag = (SegmentsStride == (Size / 2));
	//int LastStridePeriodicFlag = ((IDSegMap /(SegmentsStride<<1)) & 1) ^ ((IDSegMap /Size) & 1);
	//int LastDir = (dir ^ ((IDSegMap /(Size>> SizeChangeFlag)) & 1) )^ (LastStridePeriodicFlag & (!SizeChangeFlag));
	/*
	FlagsShift = tempDir^(Dev_TestArray[ICLevelRightHalfEnd] >= Dev_TestArray[ICLevelRightHalfStart]);

	OEFlag = ((ICLevelEnd - ICLevelStart + 1)%2 != 0)&FlagsShift;
	*/

	Stride = (ICLevelEnd - ICLevelStart + 1)/2 + OEFlags[IDSegMap];

	//Stride = (ICLevelEnd - ICLevelStart + 1) / 2 + Shared_OEFlag;

	
	if(pos<= ICSegEnd &&  (pos + Stride) <= ICLevelEnd){

		/*
		Left = Dev_TestArray[pos];
		Right = Dev_TestArray[pos + Stride];

		Comparetor(Left, Right, tempDir);

		Dev_TestArray[pos] = Left;
		Dev_TestArray[pos + Stride] = Right;
		*/

		Comparetor(Dev_TestArray[pos], Dev_TestArray[pos + Stride], tempDir);
	}
	
}

__global__ void TestKernel(int TotalSegments,int* TestArray, int** InArray) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int TheValLeft = 0;
	int TheValRight = 0;

	if (cid < TotalSegments) {
		TheValLeft = *(*(InArray + cid) + 0);
		TheValRight = *(*(InArray + cid) + 1);

		//printf("Hello thread %d\n", threadIdx.x);

		TestArray[2 * cid] = TheValLeft;
		TestArray[2 * cid + 1] = TheValRight;
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge(double* Dev_TestArray, int** IDStartEnd, int dir, double padNum, int Size) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int LastPowerTwo;
	int SegmentSize;
	int ICStart = IDStartEnd[bid][0];
	int ICEnd = IDStartEnd[bid][1];
	int IDRelative = ICStart + tid;
	double tempPadNum = (1 - 2* ((bid / Size) % 2))*padNum;

	double __shared__ Share_TestArray[BLOCKSIZE];

	Share_TestArray[tid] = tempPadNum;
	if (IDRelative <= ICEnd) {
		Share_TestArray[tid] = Dev_TestArray[IDRelative];
	}

	Share_TestArray[tid + BLOCKSIZE / 2] = tempPadNum;
	if ((IDRelative + BLOCKSIZE / 2) <= ICEnd) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[IDRelative + BLOCKSIZE / 2];
	}

	LastPowerTwo = 1;

	SegmentSize = ICEnd - ICStart + 1;

	for (int i = 2; i < SegmentSize; i <<= 1) {

		LastPowerTwo = i;

		tempDir = ((bid/ Size) % 2) ^ (dir ^ ((tid & (i / 2)) != 0));

		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * tid - (tid & (stride - 1));

			//if ((pos + stride) < NSize) {
			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			//}

		}
	}

	tempDir = ((bid / Size) % 2) ^dir;

	for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * tid - (tid & (stride - 1));

		//if ((pos + stride) < NSize) {
		Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
		//}

	}

	__syncthreads();

	if (IDRelative <= ICEnd) {
		Dev_TestArray[IDRelative] = Share_TestArray[tid];
	}
	if ((IDRelative + BLOCKSIZE / 2) <= ICEnd) {
		Dev_TestArray[IDRelative + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge_Last(double* Dev_TestArray, int** IDStartEnd, int dir, int Size) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int ICStart = IDStartEnd[bid][0];
	int ICEnd = IDStartEnd[bid][1];
	int IDRelative = ICStart + tid;
	int stride;
	int tempSegmentSize;
	int tempICStart;
	int tempICEnd;
	int LastSegmentSize;
	int LastRemindSegmentSize;
	int tempAllSize;
	int FlagsShift;
	int OEFlags;
	int FlagOne;
	int tempLevelSeg;
	int LRFlags;

	double __shared__ Share_TestArray[BLOCKSIZE];

	if (IDRelative <= ICEnd) {
		Share_TestArray[tid] = Dev_TestArray[IDRelative];
	}

	if ((IDRelative + BLOCKSIZE / 2) <= ICEnd) {
		Share_TestArray[tid + BLOCKSIZE / 2] = Dev_TestArray[IDRelative + BLOCKSIZE / 2];
	}

	tempDir = ((bid / Size)%2) ^ dir;
	tempSegmentSize = ICEnd - ICStart + 1;
	tempICStart = 0;
	tempICEnd = tempICStart + tempSegmentSize/2 - 1;
	LastSegmentSize = tempSegmentSize;
	LastRemindSegmentSize = LastSegmentSize - LastSegmentSize / 2;
	tempAllSize = ICEnd - ICStart + 1;
	OEFlags = 0;
	LRFlags = 0;  // 0 for Left , 1 for Right

	for (int LevelSeg = BLOCKSIZE/2; LevelSeg > 0; LevelSeg >>= 1) {

		__syncthreads();

		FlagOne = (1 == LevelSeg) & (tempAllSize == 3);

		tempLevelSeg = LevelSeg * (!FlagOne) + 2 * FlagOne;

		LRFlags = (tid / tempLevelSeg) % 2;

		tempAllSize = (LastSegmentSize * (LRFlags ^ 1) + LastRemindSegmentSize * (LRFlags & 1))*(!FlagOne) + tempAllSize* FlagOne;

		tempSegmentSize = tempAllSize / 2;

		tempICStart = tempICStart + (LastSegmentSize * (LRFlags & 1))*(!FlagOne) + FlagOne;
		tempICEnd = tempICStart + tempSegmentSize - 1;

		LastRemindSegmentSize = tempAllSize - tempAllSize / 2;

		LastSegmentSize = tempSegmentSize;

		FlagsShift = tempDir ^ (Share_TestArray[tempICEnd + LastRemindSegmentSize] >= Share_TestArray[tempICEnd + 1]);

		OEFlags = FlagsShift & ((tempAllSize % 2) != 0);

		pos = tempICStart + tid - (tid / tempLevelSeg)*tempLevelSeg;

		stride = tempSegmentSize + OEFlags* (!FlagOne);

		__syncthreads();

		if (pos <= tempICEnd) {

			Comparetor(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);

		}

	}
	
	__syncthreads();

	if (IDRelative <= ICEnd) {
		Dev_TestArray[IDRelative] = Share_TestArray[tid];
	}
	if ((IDRelative + BLOCKSIZE / 2) <= ICEnd) {
		Dev_TestArray[IDRelative + BLOCKSIZE / 2] = Share_TestArray[tid + BLOCKSIZE / 2];
	}
}


extern "C" void FillTheSEArray(int Level, int Left, int Right, int Index, int** SEArray) {

	if (0 == Level || Left == Right) {

		SEArray[Index][0] = Left;

		SEArray[Index][1] = Right;

		return;
	}

	FillTheSEArray(Level-1, Left, Left + (Right - Left + 1)/2 - 1, Index*2, SEArray);
	FillTheSEArray(Level-1, Left + (Right - Left + 1) / 2, Right, Index*2 + 1, SEArray);
}



/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
void CPU_Shared_Merge(double* Host_TestArray, int** IDStartEnd, int dir, int Size, int TotalSegments) {
	
	for (int bid = 0; bid < TotalSegments; bid++) {
		int tempDir = ((bid / Size) % 2) ^ dir;

		int SegmentSize;
		int ICStart = IDStartEnd[bid][0];
		int ICEnd = IDStartEnd[bid][1];

		for (int Loop = ICStart; Loop <= ICEnd; Loop++){
			for (int cid = ICStart; cid < ICEnd; cid++) {
				Comparetor_Host(Host_TestArray[cid], Host_TestArray[cid + 1], tempDir);
			}
		}

	}
}

extern "C" void CPU_GlobalMerge(int Size, int TotalSegments, int SegmentsStride, int MaxSegments, int** IDStartEnd, double* Host_TestArray, int dir) {
	int tempDir;
	int pos;
	double Left;
	double Right;

	for (int IDSeg = 0; IDSeg < TotalSegments/2; IDSeg++) {
		int IDLevel = IDSeg / SegmentsStride;
		int IDSegStart = IDLevel * 2 * SegmentsStride;
		int IDSegEnd = (IDLevel + 1) * 2 * SegmentsStride - 1;
		int IDSegMap = IDSegStart + IDSeg % SegmentsStride;
		int ICSegStart;
		int ICSegEnd;
		int ICLevelStart;
		int ICLevelRightHalfStart;
		int ICLevelRightHalfEnd;
		int ICLevelEnd;
		int FlagsShift;
		int Stride;
		int OEFlag; // odd or Even within size;

		tempDir = dir ^ ((IDSegStart / Size) & 1);

		//ICSegStart = IDStartEnd[IDSegMap][0];
		//ICSegEnd = IDStartEnd[IDSegMap][1];
		ICSegStart = *(*(IDStartEnd + IDSegMap) + 0);
		ICSegEnd = *(*(IDStartEnd + IDSegMap) + 1);


		//ICLevelStart = IDStartEnd[IDSegStart][0];
		//ICLevelEnd = IDStartEnd[IDSegEnd][1];
		ICLevelStart = *(*(IDStartEnd + IDSegStart) + 0);
		ICLevelEnd = *(*(IDStartEnd + IDSegEnd) + 1);

		//ICLevelRightHalfStart = IDStartEnd[IDSegStart + SegmentsStride][0];
		ICLevelRightHalfStart = *(*(IDStartEnd + IDSegStart + SegmentsStride) + 0);
		ICLevelRightHalfEnd = ICLevelEnd;

		//int SegmentsStrideLast = SegmentsStride >> (SegmentsStride == (Size / 2));
		//int IDSegStartLast = 2*(IDSegMap / (2*SegmentsStrideLast))*SegmentsStrideLast;
		//int MainDir = dir^(  (IDSegStartLast / (Size / (1 + (SegmentsStride == (Size/2)) )   )) & 1);

		//int SizeChangeFlag = (SegmentsStride == (Size / 2));
		//int LastStridePeriodicFlag = ((IDSegMap /(SegmentsStride<<1)) & 1) ^ ((IDSegMap /Size) & 1);
		//int LastDir = (dir ^ ((IDSegMap /(Size>> SizeChangeFlag)) & 1) )^ (LastStridePeriodicFlag & (!SizeChangeFlag));
		FlagsShift = tempDir ^ (Host_TestArray[ICLevelRightHalfEnd] >= Host_TestArray[ICLevelRightHalfStart]);

		OEFlag = ((ICLevelEnd - ICLevelStart + 1) % 2 != 0)&FlagsShift;

		Stride = (ICLevelEnd - ICLevelStart + 1) / 2 + OEFlag;

		for (int tid = 0; tid < MaxSegments; tid++) {
			pos = ICSegStart + tid;



			if (pos <= ICSegEnd && (pos + Stride) <= ICLevelEnd) {

				Left = Host_TestArray[pos];
				Right = Host_TestArray[pos + Stride];

				Comparetor_Host(Left, Right, tempDir);

				Host_TestArray[pos] = Left;
				Host_TestArray[pos + Stride] = Right;

			}

		}

	}
}


extern "C" void ArbitraryBitonicSort(int NSize, double* Host_TestArrayIn, double* Host_TestArrayOut, int dir, float & timerArbitraryBitonicSort) {
	//Local Vars
	double* Dev_TestArray;
	double* Host_TestArray;
	int *OEFlags;
	double* Dev_OutToHostArray;
	double *GPU_PreSortValue;
	double *CPU_PreSortValue;
	int NBGlobal;
	int NBXGlobal;
	int NBYGlobal;
	int BXGlobal;
	int BYGlobal;
	dim3 blocksGlobal;
	dim3 threadsGlobal;
	int NBShared;
	int NBXShared;
	int NBYShared;
	int BXShared;
	int BYShared;
	dim3 blocksShared;
	dim3 threadsShared;
	double padNum;
	int tempNSize;
	int **IDStartEnd_Host;
	int **IDStartEnd_Dev;
	int *IDStartEnd_Dev_OneDim;
	int **AddrStartEnd_HostRecordDev;
	int **TestStartEnd_Host;
	int *IDStartEnd_Dev_Test;
	int *IDStartEnd_Host_Test;
	int Level;
	int none;
	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);


	cudaError_t error = cudaMalloc((void**)&Dev_TestArray, NSize * sizeof(double));

	if (cudaSuccess != error) {
		std::cout << "Error occour when malloc " << std::endl;
		std::cin>>none;
	}

	error = cudaMemcpy(Dev_TestArray, Host_TestArrayIn, NSize * sizeof(double), cudaMemcpyHostToDevice);

	if (cudaSuccess != error) {
		std::cout << "Error occour when copy " << std::endl;
		std::cin >> none;
	}


	Host_TestArray = new double[NSize];
	Dev_OutToHostArray = new double[NSize];
	GPU_PreSortValue = new double[NSize];
	CPU_PreSortValue = new double[NSize];

	for (int i = 0; i < NSize; i++) {
		Host_TestArray[i] = Host_TestArrayIn[i];
	}

	padNum = -1.E32;

	if (0 != dir) {
		padNum = 1.E32;
	}
	

	int MaxSegments = NSize;
	int TotalSegments = 1;

	Level = 0;
	while (MaxSegments > BLOCKSIZE) {
		MaxSegments = MaxSegments - MaxSegments/2;
		TotalSegments <<= 1;
		Level++;
	}

	IDStartEnd_Host = new int*[TotalSegments];

	AddrStartEnd_HostRecordDev = new int*[TotalSegments];

	for (int i = 0; i < TotalSegments; i++) {
		IDStartEnd_Host[i] = new int[2];

		for (int j = 0; j < 2;j++) {
			IDStartEnd_Host[i][j] = 0;
		}
	}


	cudaMalloc((void**)&OEFlags, sizeof(int)*TotalSegments);
	//IDStartEnd_Dev = new int*[TotalSegments];

	//for (int i = 0; i < TotalSegments; i++) {
	//	error = cudaMalloc((void**)&(IDStartEnd_Dev[i]), 2 * sizeof(int));
	//}
	
	//std::cout << "NSize " << NSize << " MaxSegments " << MaxSegments << " TotalSegments " << TotalSegments << std::endl;

	FillTheSEArray(Level, 0, NSize - 1,0, IDStartEnd_Host);

	/*Check*/
	int Length = 2;

	for (int i = 0; i <= Level; i++) {
		int HalfLength = Length/2;

		for (int j = 0; j< TotalSegments/ Length;j++) {
			int LeftSize = IDStartEnd_Host[j*Length + HalfLength - 1][1] - IDStartEnd_Host[j*Length][0];
			int RightSize = IDStartEnd_Host[j*Length + Length - 1][1] - IDStartEnd_Host[j*Length + HalfLength][0];

			if (LeftSize > RightSize || (RightSize - LeftSize) > 1) {
				std::cout << "The Segments divide is wrong. " << std::endl;
				std::cin >> none;
			}
		}
		
		Length <<= 1;
	}

	error = cudaMalloc((void**)&IDStartEnd_Dev, TotalSegments * sizeof(int*));

	if (cudaSuccess != error) {
		std::cout << "Error occour when malloc " << std::endl;
		std::cin >> none;
	}

	
	for (int i = 0; i < TotalSegments; i++) {
		error = cudaMalloc((void**)&IDStartEnd_Dev_OneDim, 2 * sizeof(int));

		cudaMemcpy(IDStartEnd_Dev_OneDim, IDStartEnd_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

		AddrStartEnd_HostRecordDev[i] = IDStartEnd_Dev_OneDim;
	}

	cudaMemcpy(IDStartEnd_Dev, AddrStartEnd_HostRecordDev, TotalSegments * sizeof(int*), cudaMemcpyHostToDevice);

	//for (int i = 0; i < TotalSegments; i++) {
	//	cudaMemcpy(IDStartEnd_Dev[i], IDStartEnd_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);
	//}

	//for (int i = 0; i < TotalSegments; i++) {

	//	std::cout << IDStartEnd_Host[i][0] << "   " << IDStartEnd_Host[i][1] << std::endl;
	//}

	TestStartEnd_Host = new  int*[TotalSegments];
	for (int i = 0; i < TotalSegments; i++) {
		TestStartEnd_Host[i] = new int[2];

		for (int j = 0; j < 2; j++) {
			TestStartEnd_Host[i][j] = 0;
		}
	}

	cudaMemcpy(AddrStartEnd_HostRecordDev,IDStartEnd_Dev,TotalSegments * sizeof(int*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < TotalSegments; i++) {
		cudaMemcpy(TestStartEnd_Host[i], AddrStartEnd_HostRecordDev[i], 2 * sizeof(int), cudaMemcpyDeviceToHost);
	}

	for (int i = 0; i < TotalSegments; i++) {

		if (TestStartEnd_Host[i][0] != IDStartEnd_Host[i][0] || TestStartEnd_Host[i][1] != IDStartEnd_Host[i][1]) {
			std::cout << "Wrong ... in index : " << i << std::endl;
			std::cout << TestStartEnd_Host[i][0] << " " << TestStartEnd_Host[i][1] << " " << IDStartEnd_Host[i][0] << " " << IDStartEnd_Host[i][1] << std::endl;
			std::cin >> none;
		}

		if ((IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0] + 1) > BLOCKSIZE) {
			std::cout << "Wrong ... in index that segments greather than BLOCKSIZE: " << i << std::endl;
			std::cout << IDStartEnd_Host[i][0] << " " << IDStartEnd_Host[i][1] << std::endl;
			std::cin >> none;
		}

		if ((IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) < 0) {
			std::cout << "Wrong ... in index that segments less than 0: " << i << std::endl;
			std::cout << IDStartEnd_Host[i][0] << " " << IDStartEnd_Host[i][1] << std::endl;
			std::cin >> none;
		}

		if (i > 0) {
			if ( abs( ((IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) - (IDStartEnd_Host[i - 1][1] - IDStartEnd_Host[i - 1][0])) ) > 1) {
				std::cout << "Wrong ... in index that neighbor Segments cannot greater than 1: " << i << std::endl;
				std::cout << IDStartEnd_Host[i-1][0] << " " << IDStartEnd_Host[i-1][1] << std::endl;
				std::cout << IDStartEnd_Host[i][0] << " " << IDStartEnd_Host[i][1] << std::endl;
				std::cin >> none;
			}
		}
	}

	
	cudaMalloc((void**)&IDStartEnd_Dev_Test, TotalSegments * 2 * sizeof(int));

	if (cudaSuccess != error) {
		std::cout << "Error occour when malloc " << std::endl;
		std::cin >> none;
	}

	IDStartEnd_Host_Test = new int[TotalSegments * 2];

	cudaDeviceSynchronize();

	int NBTest = TotalSegments / BLOCKSIZE + 1;
	int NBXTest = BLOCKSIZE;
	int NBYTest = 1;
	dim3 blocksTest = dim3(NBTest,1,1);
	dim3 threadsTest = dim3(NBXTest, NBYTest,1);

	TestKernel << <blocksTest, threadsTest >> > (TotalSegments,IDStartEnd_Dev_Test, IDStartEnd_Dev);

	cudaDeviceSynchronize();

	error = cudaGetLastError();

	if (cudaSuccess != error) {
		std::cout << "Error occour when call kernel TestKernel " << std::endl;

		std::cout << cudaGetErrorName(error)<< std::endl;
		std::cout << cudaGetErrorString(error) << std::endl;
		std::cin >> none;
	}

	cudaDeviceSynchronize();

	cudaMemcpy(IDStartEnd_Host_Test, IDStartEnd_Dev_Test,2*TotalSegments * sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < TotalSegments; i++) {

		if (IDStartEnd_Host_Test[2*i] != IDStartEnd_Host[i][0] || IDStartEnd_Host_Test[2 * i + 1] != IDStartEnd_Host[i][1]) {
			std::cout << "Wrong ... in index KK : " << i << std::endl;
			std::cout << IDStartEnd_Host_Test[2 * i] << " " << IDStartEnd_Host_Test[2 * i + 1] << " " << IDStartEnd_Host[i][0] << " " << IDStartEnd_Host[i][1] << std::endl;
			std::cin >> none;
		}

		//std::cout << IDStartEnd_Host_Test[2 * i] << "   " << IDStartEnd_Host_Test[2 * i + 1] << std::endl;
	}

	BXGlobal = BLOCKSIZE;
	BYGlobal = 1;
	NBGlobal = TotalSegments / 2;
	NBXGlobal = NBGlobal;
	NBYGlobal = 1;
	blocksGlobal = dim3(NBXGlobal, NBYGlobal, 1);
	threadsGlobal = dim3(BXGlobal, BYGlobal, 1);

	BXShared = BLOCKSIZE / 2;
	BYShared = 1;
	NBShared = TotalSegments;
	NBXShared = NBShared;
	NBYShared = 1;
	blocksShared = dim3(NBXShared, NBYShared, 1);
	threadsShared = dim3(BXShared, BYShared, 1);

	cudaDeviceSynchronize();

	cudaEventRecord(StartEvent, 0);

	if (NSize <= BLOCKSIZE) {
		Kernel_Shared_ArbitraryBitonicSort << <blocksShared, threadsShared >> > (NSize, Dev_TestArray, dir, padNum);
	}else {

		Kernel_Shared_Merge << <blocksShared, threadsShared >> > (Dev_TestArray, IDStartEnd_Dev, dir, padNum, 1);

		for (int Size = 2; Size <= TotalSegments; Size<<=1) {

			for (int Stride = Size / 2; Stride >= 0; Stride >>= 1) {
				
				if (Stride >= 1) {

					Kernel_GlobalMerge_Pre << <blocksGlobal, 1 >> > (Size, Stride, MaxSegments, IDStartEnd_Dev, Dev_TestArray, dir, OEFlags);

					Kernel_GlobalMerge << <blocksGlobal, threadsGlobal >> > (Size, Stride, MaxSegments, IDStartEnd_Dev, Dev_TestArray, dir, OEFlags);
				}
				else {

					//Kernel_Shared_Merge<< <blocksShared, threadsShared >> > (Dev_TestArray, IDStartEnd_Dev, dir, padNum, Size);

					Kernel_Shared_Merge_Last << <blocksShared, threadsShared >> > (Dev_TestArray, IDStartEnd_Dev, dir, Size);

					break;
				}
				
			}
			
		}
	}

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerArbitraryBitonicSort, StartEvent, StopEvent);

	cudaDeviceSynchronize();

	cudaMemcpy(Host_TestArrayOut, Dev_TestArray, NSize * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(Dev_TestArray);

	cudaFree(OEFlags);
	
	/*
	for (int i = 0; i < TotalSegments; i++) {
		error = cudaFree(AddrStartEnd_HostRecordDev[i]);
	}
	cudaFree(IDStartEnd_Dev_OneDim);
	error = cudaFree(IDStartEnd_Dev);
	*/
	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);
}