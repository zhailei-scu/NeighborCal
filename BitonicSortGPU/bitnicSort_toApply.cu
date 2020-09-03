#include "bitnicSort_toApply.h"
#include <iostream>

#define BLOCKSIZE_TOAPPLY 1024

__device__ inline void Comparetor_toApply(double &ValA, double &ValB, int dir) {
	double temp;

	// dir 0 for decreasing , dir 1 for increasing

	if ((ValA > ValB) == dir) {
		temp = ValA;
		ValA = ValB;
		ValB = temp;
	}
}

inline void Comparetor_Host_toApply(double &ValA, double &ValB, int dir) {
	double temp;

	// dir 0 for decreasing , dir 1 for increasing

	if ((ValA > ValB) == dir) {
		temp = ValA;
		ValA = ValB;
		ValB = temp;
	}
}


/*Used For Array Size less than BLOCKSIZE And is of power 2*/
__global__ void Kernel_PowerTowShared_toApply(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE_TOAPPLY + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = 0.E0;
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2];
	}

	for (int i = 2; i < NSize; i <<= 1) {

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * cid - (cid & (stride - 1));

			if ((pos + stride) < NSize) {
				Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			}

		}
	}

	for (int stride = NSize; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * cid - (cid & (stride - 1));

		if ((pos + stride) < NSize) {
			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		}

	}

	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


__global__ void Kernel_ArbitraryBitonicSort_toApply(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE_TOAPPLY + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = 0.E0;
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2];
	}

	int NThreadUpHalf = NSize / 2;

	int IDRelative = cid % NThreadUpHalf;

	int IDStart = NThreadUpHalf * (cid / NThreadUpHalf);

	int IDEnd = NThreadUpHalf + (cid / NThreadUpHalf)*(NSize - NThreadUpHalf);

	int FirstGreater = (NThreadUpHalf - 1) << 1;

	//for (int i = 2; i < NThreadUpHalf; i <<= 1) {
	for (int i = 2; i <= 16; i <<= 1) {
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
				Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
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
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}

__global__ void Kernel_ArbitraryBitonicSort2_toApply(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE_TOAPPLY + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = 0.E0;
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2];
	}

	for (int i = 2; i < NSize; i <<= 1) {

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * cid - (cid & (stride - 1));

			if ((pos + stride) < NSize) {
				Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
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
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


__device__ int FirstGreaterPower2_toApply(int NSize) {
	int result;

	result = 1;

	while (result > 0 && result < NSize) {
		result <<= 1;
	}

	return result;
}

__device__ int GetIDStart_toApply(int NSize, int Stride, int cid) {
	int result;

	int NHalf;
	int TopHalfThread = FirstGreaterPower2_toApply(NSize / 2);

	NHalf = NSize / 2;
	if (cid >= TopHalfThread) {
		NHalf = NSize - NHalf;
	}

	result = cid;

	return result;
}

__global__ void Kernel_ArbitraryBitonicSort3_toApply(int NSize, double* Dev_TestArray, int dir) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE_TOAPPLY + tid;
	int tempDir;
	int pos;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = 0.E0;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = 0.E0;
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2];
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
				Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
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
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_ArbitraryBitonicSort_toApply(int NSize, double* Dev_TestArray, int dir, double padNum) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE_TOAPPLY + tid;
	int tempDir;
	int pos;
	int LastPowerTwo;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = padNum;
	if (cid < NSize) {
		Share_TestArray[tid] = Dev_TestArray[cid];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = padNum;
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2];
	}

	LastPowerTwo = 1;

	for (int i = 2; i < NSize; i <<= 1) {

		LastPowerTwo = i;

		tempDir = dir ^ ((cid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * cid - (cid & (stride - 1));

			//if ((pos + stride) < NSize) {
			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			//}

		}
	}

	for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * cid - (cid & (stride - 1));

		//if ((pos + stride) < NSize) {
		Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], dir);
		//}

	}

	__syncthreads();

	if (cid < NSize) {
		Dev_TestArray[cid] = Share_TestArray[tid];
	}
	if ((cid + BLOCKSIZE_TOAPPLY / 2) < NSize) {
		Dev_TestArray[cid + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


__global__ void Kernel_GlobalMerge_Pre_toApply(int Size, int SegmentsStride, int MaxSegments, int** IDStartEnd, double* Dev_TestArray, int dir, int *OEFlags) {
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


__global__ void Kernel_GlobalMerge_toApply(int Size, int SegmentsStride, int MaxSegments, int** IDStartEnd, double* Dev_TestArray, int dir, int *OEFlags) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int tempDir;
	int pos;
	double Left;
	double Right;
	int IDSeg = cid / BLOCKSIZE_TOAPPLY;
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
	int Stride;

	tempDir = dir ^ ((IDSegStart / Size) & 1);

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

	Stride = (ICLevelEnd - ICLevelStart + 1) / 2 + OEFlags[IDSegMap];

	//Stride = (ICLevelEnd - ICLevelStart + 1) / 2 + Shared_OEFlag;


	if (pos <= ICSegEnd && (pos + Stride) <= ICLevelEnd) {

		/*
		Left = Dev_TestArray[pos];
		Right = Dev_TestArray[pos + Stride];

		Comparetor(Left, Right, tempDir);

		Dev_TestArray[pos] = Left;
		Dev_TestArray[pos + Stride] = Right;
		*/

		Comparetor_toApply(Dev_TestArray[pos], Dev_TestArray[pos + Stride], tempDir);
	}

}



/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge_toApply(double* Dev_TestArray, int** IDStartEnd, int dir, double padNum, int Size) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int LastPowerTwo;
	int SegmentSize;
	int ICStart = IDStartEnd[bid][0];
	int ICEnd = IDStartEnd[bid][1];
	int IDRelative = ICStart + tid;
	double tempPadNum = (1 - 2 * ((bid / Size) % 2))*padNum;

	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	Share_TestArray[tid] = tempPadNum;
	if (IDRelative <= ICEnd) {
		Share_TestArray[tid] = Dev_TestArray[IDRelative];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = tempPadNum;
	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2];
	}

	LastPowerTwo = 1;

	SegmentSize = ICEnd - ICStart + 1;

	for (int i = 2; i < SegmentSize; i <<= 1) {

		LastPowerTwo = i;

		tempDir = ((bid / Size) % 2) ^ (dir ^ ((tid & (i / 2)) != 0));

		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * tid - (tid & (stride - 1));

			//if ((pos + stride) < NSize) {
			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
			//}

		}
	}

	tempDir = ((bid / Size) % 2) ^ dir;

	for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * tid - (tid & (stride - 1));

		//if ((pos + stride) < NSize) {
		Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
		//}

	}

	__syncthreads();

	if (IDRelative <= ICEnd) {
		Dev_TestArray[IDRelative] = Share_TestArray[tid];
	}
	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
		Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge_Last_toApply(double* Dev_TestArray, int** IDStartEnd, int dir, int Size) {
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

	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	if (IDRelative <= ICEnd) {
		Share_TestArray[tid] = Dev_TestArray[IDRelative];
	}

	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2];
	}

	tempDir = ((bid / Size) % 2) ^ dir;
	tempSegmentSize = ICEnd - ICStart + 1;
	tempICStart = 0;
	tempICEnd = tempICStart + tempSegmentSize / 2 - 1;
	LastSegmentSize = tempSegmentSize;
	LastRemindSegmentSize = LastSegmentSize - LastSegmentSize / 2;
	tempAllSize = ICEnd - ICStart + 1;
	OEFlags = 0;
	LRFlags = 0;  // 0 for Left , 1 for Right

	for (int LevelSeg = BLOCKSIZE_TOAPPLY / 2; LevelSeg > 0; LevelSeg >>= 1) {

		__syncthreads();

		FlagOne = (1 == LevelSeg) & (tempAllSize == 3);

		tempLevelSeg = LevelSeg * (!FlagOne) + 2 * FlagOne;

		LRFlags = (tid / tempLevelSeg) % 2;

		tempAllSize = (LastSegmentSize * (LRFlags ^ 1) + LastRemindSegmentSize * (LRFlags & 1))*(!FlagOne) + tempAllSize * FlagOne;

		tempSegmentSize = tempAllSize / 2;

		tempICStart = tempICStart + (LastSegmentSize * (LRFlags & 1))*(!FlagOne) + FlagOne;
		tempICEnd = tempICStart + tempSegmentSize - 1;

		LastRemindSegmentSize = tempAllSize - tempAllSize / 2;

		LastSegmentSize = tempSegmentSize;

		FlagsShift = tempDir ^ (Share_TestArray[tempICEnd + LastRemindSegmentSize] >= Share_TestArray[tempICEnd + 1]);

		OEFlags = FlagsShift & ((tempAllSize % 2) != 0);

		pos = tempICStart + tid - (tid / tempLevelSeg)*tempLevelSeg;

		stride = tempSegmentSize + OEFlags * (!FlagOne);

		__syncthreads();

		if (pos <= tempICEnd) {

			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);

		}

	}

	__syncthreads();

	if (IDRelative <= ICEnd) {
		Dev_TestArray[IDRelative] = Share_TestArray[tid];
	}
	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
		Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


extern "C" void FillTheSEArray_toApply(int Level, int Left, int Right, int Index, int** SEArray) {

	if (0 == Level || Left == Right) {

		SEArray[Index][0] = Left;

		SEArray[Index][1] = Right;

		return;
	}

	FillTheSEArray_toApply(Level - 1, Left, Left + (Right - Left + 1) / 2 - 1, Index * 2, SEArray);
	FillTheSEArray_toApply(Level - 1, Left + (Right - Left + 1) / 2, Right, Index * 2 + 1, SEArray);
}



/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
void CPU_Shared_Merge_toApply(double* Host_TestArray, int** IDStartEnd, int dir, int Size, int TotalSegments) {

	for (int bid = 0; bid < TotalSegments; bid++) {
		int tempDir = ((bid / Size) % 2) ^ dir;

		int SegmentSize;
		int ICStart = IDStartEnd[bid][0];
		int ICEnd = IDStartEnd[bid][1];

		for (int Loop = ICStart; Loop <= ICEnd; Loop++) {
			for (int cid = ICStart; cid < ICEnd; cid++) {
				Comparetor_Host_toApply(Host_TestArray[cid], Host_TestArray[cid + 1], tempDir);
			}
		}

	}
}

extern "C" void CPU_GlobalMerge_toApply(int Size, int TotalSegments, int SegmentsStride, int MaxSegments, int** IDStartEnd, double* Host_TestArray, int dir) {
	int tempDir;
	int pos;
	double Left;
	double Right;

	for (int IDSeg = 0; IDSeg < TotalSegments / 2; IDSeg++) {
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

				Comparetor_Host_toApply(Left, Right, tempDir);

				Host_TestArray[pos] = Left;
				Host_TestArray[pos + Stride] = Right;

			}

		}

	}
}

extern "C" void ArbitraryBitonicSort_toApply(int NSize, double* ToSortDev_ClustersPosX,int* SortedIndex, int dir) {
	//Local Vars
	int *OEFlags;
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
	int Level;
	int **IDStartEnd_Host;
	int **IDStartEnd_Dev;
	int *IDStartEnd_Dev_OneDim;
	int **AddrStartEnd_HostRecordDev;
	int error;

	padNum = -1.E32;

	if (0 != dir) {
		padNum = 1.E32;
	}

	int MaxSegments = NSize;
	int TotalSegments = 1;

	Level = 0;
	while (MaxSegments > BLOCKSIZE_TOAPPLY) {
		MaxSegments = MaxSegments - MaxSegments / 2;
		TotalSegments <<= 1;
		Level++;
	}

	IDStartEnd_Host = new int*[TotalSegments];

	AddrStartEnd_HostRecordDev = new int*[TotalSegments];

	for (int i = 0; i < TotalSegments; i++) {
		IDStartEnd_Host[i] = new int[2];

		for (int j = 0; j < 2; j++) {
			IDStartEnd_Host[i][j] = 0;
		}
	}

	cudaMalloc((void**)&OEFlags, sizeof(int)*TotalSegments);

	FillTheSEArray_toApply(Level, 0, NSize - 1, 0, IDStartEnd_Host);

	error = cudaMalloc((void**)&IDStartEnd_Dev, TotalSegments * sizeof(int*));


	for (int i = 0; i < TotalSegments; i++) {
		error = cudaMalloc((void**)&IDStartEnd_Dev_OneDim, 2 * sizeof(int));

		cudaMemcpy(IDStartEnd_Dev_OneDim, IDStartEnd_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

		AddrStartEnd_HostRecordDev[i] = IDStartEnd_Dev_OneDim;
	}

	cudaMemcpy(IDStartEnd_Dev, AddrStartEnd_HostRecordDev, TotalSegments * sizeof(int*), cudaMemcpyHostToDevice);


	BXGlobal = BLOCKSIZE_TOAPPLY;
	BYGlobal = 1;
	NBGlobal = TotalSegments / 2;
	NBXGlobal = NBGlobal;
	NBYGlobal = 1;
	blocksGlobal = dim3(NBXGlobal, NBYGlobal, 1);
	threadsGlobal = dim3(BXGlobal, BYGlobal, 1);

	BXShared = BLOCKSIZE_TOAPPLY / 2;
	BYShared = 1;
	NBShared = TotalSegments;
	NBXShared = NBShared;
	NBYShared = 1;
	blocksShared = dim3(NBXShared, NBYShared, 1);
	threadsShared = dim3(BXShared, BYShared, 1);

	if (NSize <= BLOCKSIZE_TOAPPLY) {
		Kernel_Shared_ArbitraryBitonicSort_toApply << <blocksShared, threadsShared >> > (NSize, ToSortDev_ClustersPosX, dir, padNum);
	}
	else {

		Kernel_Shared_Merge_toApply << <blocksShared, threadsShared >> > (ToSortDev_ClustersPosX, IDStartEnd_Dev, dir, padNum, 1);

		for (int Size = 2; Size <= TotalSegments; Size <<= 1) {

			for (int Stride = Size / 2; Stride >= 0; Stride >>= 1) {

				if (Stride >= 1) {

					Kernel_GlobalMerge_Pre_toApply << <blocksGlobal, 1 >> > (Size, Stride, MaxSegments, IDStartEnd_Dev, ToSortDev_ClustersPosX, dir, OEFlags);

					Kernel_GlobalMerge_toApply << <blocksGlobal, threadsGlobal >> > (Size, Stride, MaxSegments, IDStartEnd_Dev, ToSortDev_ClustersPosX, dir, OEFlags);
				}
				else {

					Kernel_Shared_Merge_Last_toApply << <blocksShared, threadsShared >> > (ToSortDev_ClustersPosX, IDStartEnd_Dev, dir, Size);

					break;
				}

			}

		}
	}

	cudaFree(OEFlags);
}