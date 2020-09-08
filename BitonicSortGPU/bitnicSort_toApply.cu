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

/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_ArbitraryBitonicSort_toApply(double* Dev_TestArray, int** IDStartEnd_ForSort, int dir, double padNum) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int LastPowerTwo;
	int ICStart;
	int ICEnd;
	int IDRelative;
	int SegmentSize;
	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	ICStart = IDStartEnd_ForSort[bid][0];
	ICEnd = IDStartEnd_ForSort[bid][1];
	IDRelative = ICStart + tid;

	Share_TestArray[tid] = padNum;
	if (IDRelative < ICEnd) {
		Share_TestArray[tid] = Dev_TestArray[IDRelative];
	}

	Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = padNum;
	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) < ICEnd) {
		Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2];
	}

	LastPowerTwo = 1;

	SegmentSize = ICEnd - ICStart + 1;

	for (int i = 2; i < SegmentSize; i <<= 1) {

		LastPowerTwo = i;

		tempDir = dir ^ ((tid & (i / 2)) != 0);


		for (int stride = i / 2; stride > 0; stride >>= 1) {

			__syncthreads();

			pos = 2 * tid - (tid & (stride - 1));

			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);

		}
	}

	for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
		__syncthreads();

		pos = 2 * tid - (tid & (stride - 1));

		Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], dir);

	}

	__syncthreads();

	if (IDRelative < ICEnd) {
		Dev_TestArray[IDRelative] = Share_TestArray[tid];
	}
	if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) < ICEnd) {
		Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
	}
}


__global__ void Kernel_GlobalMerge_Pre_toApply(int BlockNumEachBox,int Size, int SegmentsStride, int** IDStartEnd_ForSort, double* Dev_TestArray, int dir, int *OEFlags) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int tempDir;
	int IDSegRelative;
	int IBox;
	int cid0;
	int IDSegStartRelative;
	int IDLevel;
	int IDSegStart;
	int IDSegEnd;
	int IDSegMap;
	int ICLevelStart;
	int ICLevelRightHalfStart;
	int ICLevelRightHalfEnd;
	int ICLevelEnd;
	int FlagsShift;

	IBox = cid / BlockNumEachBox;
	cid0 = IBox * BlockNumEachBox;
	IDSegRelative = cid - cid0;
	
	IDLevel = IDSegRelative / SegmentsStride;
	IDSegStartRelative = IDLevel * 2 * SegmentsStride;
	IDSegStart = 2* cid0 + IDSegStartRelative;
	IDSegEnd = 2* cid0 + (IDLevel + 1) * 2 * SegmentsStride - 1;
	IDSegMap = IDSegStart + IDSegRelative % SegmentsStride;


	tempDir = dir ^ ((IDSegStartRelative / Size) & 1);

	ICLevelStart = *(*(IDStartEnd_ForSort + IDSegStart) + 0);
	ICLevelEnd = *(*(IDStartEnd_ForSort + IDSegEnd) + 1);

	ICLevelRightHalfStart = *(*(IDStartEnd_ForSort + IDSegStart + SegmentsStride) + 0);
	ICLevelRightHalfEnd = ICLevelEnd;

	FlagsShift = tempDir ^ (Dev_TestArray[ICLevelRightHalfEnd] >= Dev_TestArray[ICLevelRightHalfStart]);

	OEFlags[IDSegMap] = ((ICLevelEnd - ICLevelStart + 1) % 2 != 0)&FlagsShift;
}


__global__ void Kernel_GlobalMerge_toApply(int BlockNumEachBox,int Size, int SegmentsStride, int** IDStartEnd_ForSort, double* Dev_TestArray, int dir, int *OEFlags) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * blockDim.x * blockDim.y + tid;
	int IBox;
	int bid0;
	int tempDir;
	int pos;
	int IDLevel;
	int IDSegStart;
	int IDSegEnd;
	int IDSegMap;
	int ICSegStart;
	int ICSegEnd;
	int ICLevelStart;
	int ICLevelEnd;
	int Stride;
	int IDSegRelative;
	int IDSegStartRelative;

	IBox = bid / BlockNumEachBox;
	bid0 = IBox * BlockNumEachBox;
	IDSegRelative = bid - bid0;
	IDLevel = IDSegRelative / SegmentsStride;
	IDSegStartRelative = IDLevel * 2 * SegmentsStride;
	IDSegStart = 2*bid0 + IDSegStartRelative;
	IDSegEnd = 2*bid0 + (IDLevel + 1) * 2 * SegmentsStride - 1;
	IDSegMap = IDSegStart + IDSegRelative % SegmentsStride;

	tempDir = dir ^ ((IDSegStartRelative / Size) & 1);

	ICSegStart = *(*(IDStartEnd_ForSort + IDSegMap) + 0);
	ICSegEnd = *(*(IDStartEnd_ForSort + IDSegMap) + 1);

	pos = ICSegStart + tid;

	ICLevelStart = *(*(IDStartEnd_ForSort + IDSegStart) + 0);
	ICLevelEnd = *(*(IDStartEnd_ForSort + IDSegEnd) + 1);

	Stride = (ICLevelEnd - ICLevelStart + 1) / 2 + OEFlags[IDSegMap];

	if (pos <= ICSegEnd && (pos + Stride) <= ICLevelEnd) {
		Comparetor_toApply(Dev_TestArray[pos], Dev_TestArray[pos + Stride], tempDir);
	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge_toApply(int BlockNumEachBox_Share,double* Dev_TestArray, int** IDStartEnd_ForSort, int dir, double padNum, int Size) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int LastPowerTwo;
	int SegmentSize;
	int ICStart;
	int ICEnd;
	int IDRelative;
	double tempPadNum;
	int IBox;
	int bid0;
	int IDSegRelative;

	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	IBox = bid / BlockNumEachBox_Share;
	bid0 = IBox * BlockNumEachBox_Share;
	IDSegRelative = bid - bid0;

	ICStart = IDStartEnd_ForSort[bid][0];
	ICEnd = IDStartEnd_ForSort[bid][1];
	IDRelative = ICStart + tid;
	tempPadNum = (1 - 2 * ((IDSegRelative / Size) % 2))*padNum;

	if (ICEnd > 0) {

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

			tempDir = ((IDSegRelative / Size) % 2) ^ (dir ^ ((tid & (i / 2)) != 0));

			for (int stride = i / 2; stride > 0; stride >>= 1) {

				__syncthreads();

				pos = 2 * tid - (tid & (stride - 1));

				Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);

			}
		}

		tempDir = ((IDSegRelative / Size) % 2) ^ dir;

		for (int stride = LastPowerTwo; stride > 0; stride >>= 1) {
			__syncthreads();

			pos = 2 * tid - (tid & (stride - 1));

			Comparetor_toApply(Share_TestArray[pos], Share_TestArray[pos + stride], tempDir);
		}

		__syncthreads();

		if (IDRelative <= ICEnd) {
			Dev_TestArray[IDRelative] = Share_TestArray[tid];
		}
		if ((IDRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
			Dev_TestArray[IDRelative + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
		}

	}
}


/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
__global__ void Kernel_Shared_Merge_Last_toApply(int BlockNumEachBox_Share,double* Dev_TestArray, int** IDStartEnd_ForSort, int dir, int Size) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int tempDir;
	int pos;
	int ICStart;
	int ICEnd;
	int ICRelative;
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
	int IBox;
	int bid0;
	int IDSegRelative;

	double __shared__ Share_TestArray[BLOCKSIZE_TOAPPLY];

	IBox = bid / BlockNumEachBox_Share;
	bid0 = IBox * BlockNumEachBox_Share;
	IDSegRelative = bid - bid0;

	ICStart = IDStartEnd_ForSort[bid][0];
	ICEnd = IDStartEnd_ForSort[bid][1];
	ICRelative = ICStart + tid;

	if (ICEnd > 0) {

		if (ICRelative <= ICEnd) {
			Share_TestArray[tid] = Dev_TestArray[ICRelative];
		}

		if ((ICRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
			Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2] = Dev_TestArray[ICRelative + BLOCKSIZE_TOAPPLY / 2];
		}

		tempDir = ((IDSegRelative / Size) % 2) ^ dir;
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

		if (ICRelative <= ICEnd) {
			Dev_TestArray[ICRelative] = Share_TestArray[tid];
		}
		if ((ICRelative + BLOCKSIZE_TOAPPLY / 2) <= ICEnd) {
			Dev_TestArray[ICRelative + BLOCKSIZE_TOAPPLY / 2] = Share_TestArray[tid + BLOCKSIZE_TOAPPLY / 2];
		}

	}
}


extern "C" void FillTheSEArray_toApply(int Level,int MaxSegmentsNumEachBox,int IBox, int Left, int Right, int Index, int** SEArray) {
	int trueIndex;

	if (0 == Level || Left == Right) {
		trueIndex = IBox*MaxSegmentsNumEachBox + Index;

		SEArray[trueIndex][0] = Left;

		SEArray[trueIndex][1] = Right;

		return;
	}

	FillTheSEArray_toApply(Level - 1, MaxSegmentsNumEachBox, IBox, Left, Left + (Right - Left + 1) / 2 - 1, Index * 2, SEArray);
	FillTheSEArray_toApply(Level - 1, MaxSegmentsNumEachBox, IBox, Left + (Right - Left + 1) / 2, Right, Index * 2 + 1, SEArray);
}



/*Used For Array Size less than BLOCKSIZE And is not of power 2*/
void CPU_Shared_Merge_toApply(double* Host_TestArray, int** IDStartEnd_ForSort, int dir, int Size, int TotalSegments) {

	for (int bid = 0; bid < TotalSegments; bid++) {
		int tempDir = ((bid / Size) % 2) ^ dir;

		int SegmentSize;
		int ICStart = IDStartEnd_ForSort[bid][0];
		int ICEnd = IDStartEnd_ForSort[bid][1];

		for (int Loop = ICStart; Loop <= ICEnd; Loop++) {
			for (int cid = ICStart; cid < ICEnd; cid++) {
				Comparetor_Host_toApply(Host_TestArray[cid], Host_TestArray[cid + 1], tempDir);
			}
		}

	}
}

extern "C" void CPU_GlobalMerge_toApply(int Size, int TotalSegments, int SegmentsStride, int MaxSegments, int** IDStartEnd_ForSort, double* Host_TestArray, int dir) {
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

		ICSegStart = *(*(IDStartEnd_ForSort + IDSegMap) + 0);
		ICSegEnd = *(*(IDStartEnd_ForSort + IDSegMap) + 1);

		ICLevelStart = *(*(IDStartEnd_ForSort + IDSegStart) + 0);
		ICLevelEnd = *(*(IDStartEnd_ForSort + IDSegEnd) + 1);

		ICLevelRightHalfStart = *(*(IDStartEnd_ForSort + IDSegStart + SegmentsStride) + 0);
		ICLevelRightHalfEnd = ICLevelEnd;

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

extern "C" void ArbitraryBitonicSort_toApply(int NBox, int** IDStartEnd_ForBox_Host, int** IDStartEnd_ForBox_Dev, double* ToSortDev_ClustersPosX, int* SortedIndex, int dir) {
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
	int **IDStartEnd_ForSort_Host;
	int **IDStartEnd_ForSort_Dev;
	int *IDStartEnd_ForSort_Dev_OneDim;
	int **AddrStartEnd_HostRecordDev;
	int error;
	int MaxSegmentsEachBox;
	int tempMaxSegmentsNumEachBox;
	int MaxSegmentsNumEachBox;
	int MaxSegmentsNumAllBox;
	int MaxClusterNumEachBox;

	padNum = -1.E32;

	if (0 != dir) {
		padNum = 1.E32;
	}


	MaxClusterNumEachBox = 0;
	MaxSegmentsNumEachBox = 0;

	for (int IBox = 0; IBox < NBox; IBox++) {
		MaxSegmentsEachBox = IDStartEnd_ForBox_Host[IBox][1] - IDStartEnd_ForBox_Host[IBox][0] + 1;
		tempMaxSegmentsNumEachBox = 1;

		if (MaxClusterNumEachBox < MaxSegmentsEachBox) MaxClusterNumEachBox = MaxSegmentsEachBox;

		Level = 0;
		while (MaxSegmentsEachBox > BLOCKSIZE_TOAPPLY) {
			MaxSegmentsEachBox = MaxSegmentsEachBox - MaxSegmentsEachBox / 2;
			tempMaxSegmentsNumEachBox <<= 1;
			Level++;
		}

		if (MaxSegmentsNumEachBox < tempMaxSegmentsNumEachBox) MaxSegmentsNumEachBox = tempMaxSegmentsNumEachBox;
	}

	MaxSegmentsNumAllBox = MaxSegmentsNumEachBox * NBox;


	IDStartEnd_ForSort_Host = new int*[MaxSegmentsNumAllBox];

	AddrStartEnd_HostRecordDev = new int*[MaxSegmentsNumAllBox];

	for (int i = 0; i < MaxSegmentsNumAllBox; i++) {
		IDStartEnd_ForSort_Host[i] = new int[2];

		for (int j = 0; j < 2; j++) {
			IDStartEnd_ForSort_Host[i][j] = -1;
		}
	}

	for (int IBox = 0; IBox < NBox; IBox++) {
		MaxSegmentsEachBox = IDStartEnd_ForBox_Host[IBox][1] - IDStartEnd_ForBox_Host[IBox][0] + 1;;

		Level = 0;
		while (MaxSegmentsEachBox > BLOCKSIZE_TOAPPLY) {
			MaxSegmentsEachBox = MaxSegmentsEachBox - MaxSegmentsEachBox / 2;
			Level++;
		}

		FillTheSEArray_toApply(Level, MaxSegmentsNumEachBox,IBox, IDStartEnd_ForBox_Host[IBox][0], IDStartEnd_ForBox_Host[IBox][1], 0, IDStartEnd_ForSort_Host);
	}

	cudaMalloc((void**)&OEFlags, sizeof(int)*MaxSegmentsNumAllBox);

	error = cudaMalloc((void**)&IDStartEnd_ForSort_Dev, MaxSegmentsNumAllBox * sizeof(int*));


	for (int i = 0; i < MaxSegmentsNumAllBox; i++) {
		error = cudaMalloc((void**)&IDStartEnd_ForSort_Dev_OneDim, 2 * sizeof(int));

		cudaMemcpy(IDStartEnd_ForSort_Dev_OneDim, IDStartEnd_ForSort_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

		AddrStartEnd_HostRecordDev[i] = IDStartEnd_ForSort_Dev_OneDim;
	}

	cudaMemcpy(IDStartEnd_ForSort_Dev, AddrStartEnd_HostRecordDev, MaxSegmentsNumAllBox * sizeof(int*), cudaMemcpyHostToDevice);


	BXGlobal = BLOCKSIZE_TOAPPLY;
	BYGlobal = 1;
	NBGlobal = MaxSegmentsNumAllBox / 2;
	NBXGlobal = NBGlobal;
	NBYGlobal = 1;
	blocksGlobal = dim3(NBXGlobal, NBYGlobal, 1);
	threadsGlobal = dim3(BXGlobal, BYGlobal, 1);

	BXShared = BLOCKSIZE_TOAPPLY / 2;
	BYShared = 1;
	NBShared = MaxSegmentsNumAllBox;
	NBXShared = NBShared;
	NBYShared = 1;
	blocksShared = dim3(NBXShared, NBYShared, 1);
	threadsShared = dim3(BXShared, BYShared, 1);

	if (MaxClusterNumEachBox <= BLOCKSIZE_TOAPPLY) {
		Kernel_Shared_ArbitraryBitonicSort_toApply << <blocksShared, threadsShared >> > (ToSortDev_ClustersPosX, IDStartEnd_ForSort_Dev, dir, padNum);
	}
	else {

		Kernel_Shared_Merge_toApply << <blocksShared, threadsShared >> > (MaxSegmentsNumEachBox,ToSortDev_ClustersPosX, IDStartEnd_ForSort_Dev, dir, padNum, 1);

		for (int Size = 2; Size <= MaxSegmentsNumEachBox; Size <<= 1) {

			for (int Stride = Size / 2; Stride >= 0; Stride >>= 1) {

				if (Stride >= 1) {

					Kernel_GlobalMerge_Pre_toApply << <blocksGlobal, 1 >> > (MaxSegmentsNumEachBox,Size, Stride, IDStartEnd_ForSort_Dev, ToSortDev_ClustersPosX, dir, OEFlags);

					Kernel_GlobalMerge_toApply << <blocksGlobal, threadsGlobal >> > (MaxSegmentsNumEachBox,Size, Stride, IDStartEnd_ForSort_Dev, ToSortDev_ClustersPosX, dir, OEFlags);
				}
				else {

					Kernel_Shared_Merge_Last_toApply << <blocksShared, threadsShared >> > (MaxSegmentsNumEachBox,ToSortDev_ClustersPosX, IDStartEnd_ForSort_Dev, dir, Size);

					break;
				}

			}

		}
	}

	cudaFree(OEFlags);
}