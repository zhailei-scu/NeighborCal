#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_atomic_functions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>
#include "MyNeighborList_multipleBox.h"
#include <algorithm>
#include <map>
#include <vector>

#define BLOCKSIZE 512


bool myCompare2(std::pair<double, int> A, std::pair<double, int> B) {
	return (A.first < B.first);
}


void SimpleSort_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double* ToSortDev_ClustersPosX, int* SortedIndex_Dev, int* ReverseMap_SortedIndex_Dev) {
	double* ToSortHost_ClustersPosX;
	int* SortedIndex_Host;
	int* ReverseMap_SortedIndex_Host;
	std::vector<std::pair<double, int>> OneBox;

	std::pair<double, int> thePair;

	ToSortHost_ClustersPosX = new double[NClusters];
	SortedIndex_Host = new int[NClusters];
	ReverseMap_SortedIndex_Host = new int[NClusters];

	cudaMemcpy(ToSortHost_ClustersPosX, ToSortDev_ClustersPosX, NClusters * sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(SortedIndex_Host, SortedIndex_Dev, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	for (int IBox = 0; IBox < NBox; IBox++) {

		OneBox.clear();

		std::vector<std::pair<double, int>>().swap(OneBox);

		for (int i = IDStartEnd_Host[IBox][0]; i <= IDStartEnd_Host[IBox][1]; i++) {
			thePair.first = ToSortHost_ClustersPosX[i];
			thePair.second = SortedIndex_Host[i];
			OneBox.push_back(thePair);
		}

		std::sort(OneBox.begin(), OneBox.end(), myCompare2);

		std::vector<std::pair<double, int>>::iterator ptr = OneBox.begin();

		for (int i = IDStartEnd_Host[IBox][0]; i <= IDStartEnd_Host[IBox][1]; i++) {
			SortedIndex_Host[i] = (*ptr).second;
			ReverseMap_SortedIndex_Host[SortedIndex_Host[i]] = i;
			ptr++;
		}
	}

	cudaMemcpy(SortedIndex_Dev, SortedIndex_Host, NClusters * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(ReverseMap_SortedIndex_Dev, ReverseMap_SortedIndex_Host, NClusters * sizeof(int), cudaMemcpyHostToDevice);

}


void JumpSegmentYRange_multipleBox(int NClusters, int NBox,int **IDStartEnd_Host, int* SortedIndexX,int* ReverseMap_SortedIndexY, int XJumpStride, int** IDSESeg_Dev, int** JumpSegmentYRange_Dev) {
	int* SortedIndexX_Host;
	int* ReverseMap_SortedIndexY_host;
	int** IDSESeg_Host;
	int** JumpSegmentYRange_Host;
	int **Addr_HostRecordDev;
	int *OneDim_Dev;
	int TotalNXJumpAllBox;
	int err;

	SortedIndexX_Host = new int[NClusters];
	ReverseMap_SortedIndexY_host = new int[NClusters];
	cudaMemcpy(SortedIndexX_Host, SortedIndexX, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ReverseMap_SortedIndexY_host, ReverseMap_SortedIndexY, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	IDSESeg_Host = new int*[NBox];

	TotalNXJumpAllBox = 0;

	for (int IBox = 0; IBox < NBox; IBox++) {
		IDSESeg_Host[IBox] = new int[2];

		IDSESeg_Host[IBox][0] = TotalNXJumpAllBox;

		TotalNXJumpAllBox = TotalNXJumpAllBox + (IDStartEnd_Host[IBox][1] - IDStartEnd_Host[IBox][0])/ XJumpStride + 1;

		IDSESeg_Host[IBox][1] = TotalNXJumpAllBox - 1;
	}

	Addr_HostRecordDev = new int*[TotalNXJumpAllBox];

	for (int i = 0; i < TotalNXJumpAllBox; i++) {

		err = cudaMalloc((void**)&OneDim_Dev, 2 * sizeof(int));

		cudaMemcpy(OneDim_Dev, IDSESeg_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

		Addr_HostRecordDev[i] = OneDim_Dev;
	}

	cudaMemcpy(IDSESeg_Dev, Addr_HostRecordDev, TotalNXJumpAllBox * sizeof(int*), cudaMemcpyHostToDevice);

	JumpSegmentYRange_Host = new int*[TotalNXJumpAllBox];
	for (int i = 0; i < TotalNXJumpAllBox; i++) {
		JumpSegmentYRange_Host[i] = new int[2];

		for (int j = 0; j < 2; j++) {
			JumpSegmentYRange_Host[i][j] = -1;
		}
	}


	for (int IBox = 0; IBox < NBox; IBox++) {
		int ISegStart = IDSESeg_Host[IBox][0];
		int ISegEnd = IDSESeg_Host[IBox][1];

		for (int ISeg = ISegStart; ISeg <= ISegEnd; ISeg++) {

			int ICStart = IDStartEnd_Host[IBox][0] + (ISeg- ISegStart) * XJumpStride;
			int ICEnd = IDStartEnd_Host[IBox][0] + (ISeg - ISegStart+1) * XJumpStride - 1;

			if (ICEnd > IDStartEnd_Host[IBox][1]) ICEnd = IDStartEnd_Host[IBox][1];

			int MaxY = -1;
			int MinY = 1E16;

			for (int j = ICStart; j <= ICEnd; j++) {
				int MappedIndex = SortedIndexX_Host[j];
				int ICY = ReverseMap_SortedIndexY_host[MappedIndex];

				if (ICY > MaxY) MaxY = ICY;
				if (ICY < MinY) MinY = ICY;
			}

			JumpSegmentYRange_Host[ISeg][0] = MinY;
			JumpSegmentYRange_Host[ISeg][1] = MaxY;

		}


	}


	for (int i = 0; i < TotalNXJumpAllBox; i++) {
	
		err = cudaMalloc((void**)&OneDim_Dev, 2 * sizeof(int));

		cudaMemcpy(OneDim_Dev, JumpSegmentYRange_Host[i], 2 * sizeof(int), cudaMemcpyHostToDevice);

		Addr_HostRecordDev[i] = OneDim_Dev;
	}

	cudaMemcpy(JumpSegmentYRange_Dev, Addr_HostRecordDev, TotalNXJumpAllBox * sizeof(int*), cudaMemcpyHostToDevice);



}

__global__ void Kernel_MyNeighborListCal_SortX_multipleBox(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int LeftBound;
	int RightBound;
	int sortedID;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	bool exitFlag;
	int NRemind;
	int MapedIdex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	__shared__ double Shared_XYZ[BLOCKSIZE][3];
	__shared__ int Shared_SortedID[BLOCKSIZE];
	__shared__ int NExitedThreadsRight;
	__shared__ int NExitedThreadsLeft;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	/*Right Hand Searching*/
	exitFlag = false;

	LeftBound = scid + (bid - bid0)*BLOCKSIZE;
	if (LeftBound < scid) LeftBound = scid;
	RightBound = scid + (bid - bid0 + 1)*BLOCKSIZE -1;
	if (RightBound > ecid) RightBound = ecid;

	NRemind = RightBound - LeftBound + 1;

	NExitedThreadsRight = 0;

	if (IC <= ecid) {

		MapedIdex = SortedIndexX[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

	}

	while (LeftBound <= RightBound) {

		if (NExitedThreadsRight >= NRemind) {
			break;
		}

		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndexX[LeftBound + tid];

			Shared_SortedID[tid] = sortedID;

			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];

			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];
		}

		__syncthreads();


		if (IC <= ecid) {

			if (false == exitFlag) {
				for (int JC = LeftBound; JC <= RightBound; JC++) {
					if (JC != IC) {

						relativeJC = JC - LeftBound;

						distanceX = Shared_XYZ[relativeJC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeJC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeJC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeJC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsRight, 1);
							break;
						}

					}

				}
			}

		}

		__syncthreads();

		LeftBound = RightBound + 1;
		RightBound = LeftBound + BLOCKSIZE - 1;
		if (RightBound > ecid) RightBound = ecid;
	}

	/*Left Hand Searching*/
	exitFlag = false;

	LeftBound = scid + (bid - bid0 - 1)*BLOCKSIZE;
	if (LeftBound < scid) LeftBound = scid;
	RightBound = scid + (bid - bid0)*BLOCKSIZE - 1;
	if (RightBound > ecid) RightBound = ecid;

	NExitedThreadsLeft = 0;

	while (LeftBound <= RightBound) {

		if (NExitedThreadsLeft >= NRemind) {
			break;
		}


		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndexX[LeftBound + tid];

			Shared_SortedID[tid] = sortedID;

			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];

			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];

		}

		__syncthreads();

		if (IC <= ecid) {

			if (false == exitFlag) {
				for (int JC = RightBound; JC >= LeftBound; JC--) {
					if (JC != IC) {

						relativeJC = JC - LeftBound;

						distanceX = Shared_XYZ[relativeJC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeJC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeJC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeJC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsLeft, 1);
							break;
						}

					}

				}
			}

		}

		__syncthreads();

		RightBound = LeftBound - 1;
		LeftBound = RightBound - BLOCKSIZE + 1;
		if (LeftBound < scid) LeftBound = scid;
	}


	if (IC <= ecid) {
		Dev_NNearestNeighbor[MapedIdex] = NNID;
	}

}

__global__ void Kernel_MyNeighborListCal_SortX_multipleBox_noshare(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int MappedJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int MapedIdex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	if (IC <= ecid) {

		MapedIdex = SortedIndexX[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

		/*Right Hand Searching*/
		for (int JC = IC + 1; JC <= ecid; JC++) {

			MappedJC = SortedIndexX[JC];

			distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
			distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
			distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

			distanceX = distanceX * distanceX;
			distanceY = distanceY * distanceY;
			distanceZ = distanceZ * distanceZ;

			distance = distanceX + distanceY + distanceZ;

			if (minDistance > distance) {
				minDistance = distance;
				NNID = MappedJC;
			}

			if (distanceX > minDistance) {
				break;
			}
		}

		/*Left Hand Searching*/
		for (int JC = IC - 1; JC >= scid; JC--) {

			MappedJC = SortedIndexX[JC];

			distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
			distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
			distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

			distanceX = distanceX * distanceX;
			distanceY = distanceY * distanceY;
			distanceZ = distanceZ * distanceZ;

			distance = distanceX + distanceY + distanceZ;

			if (minDistance > distance) {
				minDistance = distance;
				NNID = MappedJC;
			}

			if (distanceX > minDistance) {
				break;
			}
		}

		Dev_NNearestNeighbor[MapedIdex] = NNID;
	}
}

__global__ void Kernel_MyNeighborListCal_SortX_multipleBox_noshare_LeftRightCohen(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int MappedJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int MapedIdex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	int JC;
	int LeftRemind;
	int RightRemind;
	int MaxRemind;
	bool flagLeftBreak;
	bool flagRightBreak;
	flagLeftBreak = false;
	flagRightBreak = false;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	if (IC <= ecid) {

		MapedIdex = SortedIndexX[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

		LeftRemind = IC - scid;
		RightRemind = ecid - IC;

		MaxRemind = RightRemind;
		if (LeftRemind > MaxRemind) MaxRemind = LeftRemind;

		for (int Shift = 1; Shift <= MaxRemind; Shift++) {

			/*Right Hand Searching*/
			if (Shift <= RightRemind && false == flagRightBreak) {
				JC = IC + Shift;

				MappedJC = SortedIndexX[JC];

				distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJC;
				}

				if (distanceX > minDistance) {
					flagRightBreak = true;

					if (true == flagLeftBreak) {
						break;
					}
				}
			}

			/*Left Hand Searching*/
			if (Shift <= LeftRemind && false == flagLeftBreak) {
				JC = IC - Shift;

				MappedJC = SortedIndexX[JC];

				distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJC;
				}

				if (distanceX > minDistance) {
					flagLeftBreak = true;

					if (true == flagRightBreak) {
						break;
					}
				}
			}

		}

		Dev_NNearestNeighbor[MapedIdex] = NNID;
	}
}

__global__ void Kernel_MyNeighborListCal_SortX_multipleBox_noshare_WithYLimit(int BlockNumEachBox, 
																			  int **IDStartEnd_ForBox_Dev,
																			  double** Dev_ClustersPosXYZ,
																			  int* SortedIndexX,
																			  int* ReverseMap_SortedIndexY,
																			  int XJumpStride,
																			  int** IDSESeg_ForJump_Dev,
																			  int** JumpSegYRange_Dev,
																			  int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int MappedJC;
	double distance;
	double minDistance;
	int minSortedYIndex;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int MapedIdex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	int SegStartID;
	int SegEndID;
	int ISeg;
	int JCStart;
	int JCEnd;


	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_ForBox_Dev[IBox][0];
	ecid = IDStartEnd_ForBox_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	minSortedYIndex = 1E16;

	if (IC <= ecid) {

		MapedIdex = SortedIndexX[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

		SegStartID = IDSESeg_ForJump_Dev[IBox][0];
		SegEndID = IDSESeg_ForJump_Dev[IBox][1];
		
		ISeg = SegStartID + (cid - bid0 * BLOCKSIZE) / XJumpStride;

		/*Right Hand Searching*/
		for (int JSeg = ISeg; JSeg <= SegEndID; JSeg++) {
			JCStart = scid + (JSeg - SegStartID)*XJumpStride;
			if (JCStart < (IC + 1)) JCStart = IC + 1;

			JCEnd = scid + (JSeg - SegStartID + 1)*XJumpStride - 1;

			MappedJC = SortedIndexX[JCStart];

			distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
			distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
			distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

			distanceX = distanceX * distanceX;
			distanceY = distanceY * distanceY;
			distanceZ = distanceZ * distanceZ;

			distance = distanceX + distanceY + distanceZ;

			if (minDistance > distance) {
				minDistance = distance;
				NNID = MappedJC;
				minSortedYIndex = ReverseMap_SortedIndexY[MappedJC];
			}

			if (distanceX > minDistance) {
				break;
			}


			if (minSortedYIndex >= JumpSegYRange_Dev[JSeg][0] || minSortedYIndex <= JumpSegYRange_Dev[JSeg][1]) {
				for (int JC = JCStart+1;JC<= JCEnd;JC++) {
					MappedJC = SortedIndexX[JC];

					distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
					distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
					distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

					distanceX = distanceX * distanceX;
					distanceY = distanceY * distanceY;
					distanceZ = distanceZ * distanceZ;

					distance = distanceX + distanceY + distanceZ;

					if (minDistance > distance) {
						minDistance = distance;
						NNID = MappedJC;
						minSortedYIndex = ReverseMap_SortedIndexY[MappedJC];
					}

					if (distanceX > minDistance) {
						break;
					}
				}




			}

		}

		/*Left Hand Searching*/
		for (int JSeg = ISeg; JSeg >= SegStartID; JSeg--) {

			JCStart = scid + (JSeg - SegStartID)*XJumpStride;

			JCEnd = scid + (JSeg - SegStartID + 1)*XJumpStride - 1;
			if (JCEnd > (IC - 1)) JCEnd = IC - 1;

			MappedJC = SortedIndexX[JCEnd];

			distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
			distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
			distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

			distanceX = distanceX * distanceX;
			distanceY = distanceY * distanceY;
			distanceZ = distanceZ * distanceZ;

			distance = distanceX + distanceY + distanceZ;

			if (minDistance > distance) {
				minDistance = distance;
				NNID = MappedJC;
				minSortedYIndex = ReverseMap_SortedIndexY[MappedJC];
			}

			if (distanceX > minDistance) {
				break;
			}

			if (minSortedYIndex >= JumpSegYRange_Dev[JSeg][0] || minSortedYIndex <= JumpSegYRange_Dev[JSeg][1]) {

				for (int JC = JCEnd-1; JC >= JCStart; JC--) {
					MappedJC = SortedIndexX[JC];

					distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
					distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
					distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

					distanceX = distanceX * distanceX;
					distanceY = distanceY * distanceY;
					distanceZ = distanceZ * distanceZ;

					distance = distanceX + distanceY + distanceZ;

					if (minDistance > distance) {
						minDistance = distance;
						NNID = MappedJC;
						minSortedYIndex = ReverseMap_SortedIndexY[MappedJC];
					}

					if (distanceX > minDistance) {
						break;
					}
				}
			}

		}

		Dev_NNearestNeighbor[MapedIdex] = NNID;
	}
}


__global__ void Kernel_MyNeighborListCal_SortX_multipleBox_noshare_LeftRightCohen_WithYLimit(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* Dev_NNearestNeighbor,double *CountEixsted, double *CountEixstedYZ) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int MappedJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int MapedIdex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	int JC;
	int LeftRemind;
	int RightRemind;
	int MaxRemind;
	bool flagLeftBreak;
	bool flagRightBreak;
	int ExistedCout;
	int ExistedYZCout;
	bool findMin;
	int TotalSearchCount;

	findMin = false;

	flagLeftBreak = false;
	flagRightBreak = false;

	ExistedCout = 0;
	ExistedYZCout = 0;

	TotalSearchCount = 0;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	if (IC <= ecid) {

		MapedIdex = SortedIndexX[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

		LeftRemind = IC - scid;
		RightRemind = ecid - IC;

		MaxRemind = RightRemind;
		if (LeftRemind > MaxRemind) MaxRemind = LeftRemind;

		for (int Shift = 1; Shift <= MaxRemind; Shift++) {

			/*Right Hand Searching*/
			if (Shift <= RightRemind && false == flagRightBreak) {
				JC = IC + Shift;

				TotalSearchCount++;

				MappedJC = SortedIndexX[JC];

				distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJC;
					findMin = true;
				}
				else if (true == findMin) {
					ExistedCout++;

					if (distanceY > minDistance || distanceZ > minDistance) {
						ExistedYZCout++;
					}
				}

				if (distanceX > minDistance) {
					flagRightBreak = true;

					if (true == flagLeftBreak) {
						break;
					}
				}
			}

			/*Left Hand Searching*/
			if (Shift <= LeftRemind && false == flagLeftBreak) {
				JC = IC - Shift;

				MappedJC = SortedIndexX[JC];

				TotalSearchCount++;

				distanceX = Dev_ClustersPosXYZ[MappedJC][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJC][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJC][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJC;

					findMin = true;
				}
				else if (true == findMin) {
					ExistedCout++;

					if (distanceY > minDistance || distanceZ > minDistance) {
						ExistedYZCout++;
					}
				}

				if (distanceX > minDistance) {
					flagLeftBreak = true;

					if (true == flagRightBreak) {
						break;
					}
				}
			}

		}

		Dev_NNearestNeighbor[MapedIdex] = NNID;

		CountEixsted[MapedIdex] = double(ExistedCout)/double(TotalSearchCount);

		CountEixstedYZ[MapedIdex] = double(ExistedYZCout) / double(TotalSearchCount);
	}
}



__global__ void Kernel_MyNeighborListCal_SortXY_multipleBox(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* SortedIndexY, int* ReverseMap_SortedIndexY, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int LeftBound;
	int RightBound;
	int sortedID;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	bool exitFlag;
	int NRemind;
	int MapedIndex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int ICX;
	int JCX;
	int ICY;
	int JCY;
	int MappedIndexTemp;
	__shared__ double Shared_XYZ[BLOCKSIZE][3];
	__shared__ int Shared_SortedID[BLOCKSIZE];
	__shared__ int NExitedThreadsRight;
	__shared__ int NExitedThreadsLeft;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	ICX = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	/*Right Hand Searching*/
	exitFlag = false;

	LeftBound = scid + (bid - bid0)*BLOCKSIZE;
	if (LeftBound < scid) LeftBound = scid;
	RightBound = scid + (bid - bid0 + 1)*BLOCKSIZE - 1;
	if (RightBound > ecid) RightBound = ecid;

	NRemind = RightBound - LeftBound + 1;

	NExitedThreadsRight = 0;

	if (ICX <= ecid) {

		MapedIndex = SortedIndexX[ICX];

		Pos_X = Dev_ClustersPosXYZ[MapedIndex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIndex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIndex][2];

		ICY = ReverseMap_SortedIndexY[MapedIndex];
	}

	while (LeftBound <= RightBound) {

		if (NExitedThreadsRight >= NRemind) {
			break;
		}

		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndexX[LeftBound + tid];
			Shared_SortedID[tid] = sortedID;
			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];
			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];
		}

		__syncthreads();


		if (ICX <= ecid) {

			if (false == exitFlag) {
				for (JCX = LeftBound; JCX <= RightBound; JCX++) {
					if (JCX != ICX) {

						relativeJC = JCX - LeftBound;


						//X restrict
						distanceX = Shared_XYZ[relativeJC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeJC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeJC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeJC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsRight, 1);
							break;
						}


						////Y restrict
						//JCY = ICY + JCX - ICX;
						//if (JCY < scid) JCY = scid;
						//if (JCY > ecid) JCY = ecid;
						//MappedIndexTemp = SortedIndexY[JCY];

						//distanceX = Dev_ClustersPosXYZ[MappedIndexTemp][0] - Pos_X;
						//distanceY = Dev_ClustersPosXYZ[MappedIndexTemp][1] - Pos_Y;
						//distanceZ = Dev_ClustersPosXYZ[MappedIndexTemp][2] - Pos_Z;

						//distanceX = distanceX * distanceX;
						//distanceY = distanceY * distanceY;
						//distanceZ = distanceZ * distanceZ;

						//distance = distanceX + distanceY + distanceZ;

						//if (minDistance > distance) {
						//	minDistance = distance;
						//	NNID = MappedIndexTemp;
						//}

						//if (distanceY > minDistance) {
						//	//exitFlag = true;
						//	//atomicAdd_block(&NExitedThreadsRight, 1);
						//	//break;
						//}

					}

				}
			}

		}

		__syncthreads();

		LeftBound = RightBound + 1;
		RightBound = LeftBound + BLOCKSIZE - 1;
		if (RightBound > ecid) RightBound = ecid;
	}

	/*Left Hand Searching*/
	exitFlag = false;

	LeftBound = scid + (bid - bid0 - 1)*BLOCKSIZE;
	if (LeftBound < scid) LeftBound = scid;
	RightBound = scid + (bid - bid0)*BLOCKSIZE - 1;
	if (RightBound > ecid) RightBound = ecid;

	NExitedThreadsLeft = 0;

	while (LeftBound <= RightBound) {

		if (NExitedThreadsLeft >= NRemind) {
			break;
		}


		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndexX[LeftBound + tid];
			Shared_SortedID[tid] = sortedID;
			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[sortedID][0];
			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[sortedID][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[sortedID][2];
		}

		__syncthreads();

		if (ICX <= ecid) {

			if (false == exitFlag) {
				for (JCX = RightBound; JCX >= LeftBound; JCX--) {
					if (JCX != ICX) {

						relativeJC = JCX - LeftBound;

						//X restrict
						distanceX = Shared_XYZ[relativeJC][0] - Pos_X;
						distanceY = Shared_XYZ[relativeJC][1] - Pos_Y;
						distanceZ = Shared_XYZ[relativeJC][2] - Pos_Z;

						distanceX = distanceX * distanceX;
						distanceY = distanceY * distanceY;
						distanceZ = distanceZ * distanceZ;

						distance = distanceX + distanceY + distanceZ;

						if (minDistance > distance) {
							minDistance = distance;
							NNID = Shared_SortedID[relativeJC];
						}

						if (distanceX > minDistance) {
							exitFlag = true;
							atomicAdd_block(&NExitedThreadsRight, 1);
							break;
						}


						////Y restrict
						//JCY = ICY + JCX - ICX;
						//if (JCY < scid) JCY = scid;
						//if (JCY > ecid) JCY = ecid;
						//MappedIndexTemp = SortedIndexY[JCY];

						//distanceX = Dev_ClustersPosXYZ[MappedIndexTemp][0] - Pos_X;
						//distanceY = Dev_ClustersPosXYZ[MappedIndexTemp][1] - Pos_Y;
						//distanceZ = Dev_ClustersPosXYZ[MappedIndexTemp][2] - Pos_Z;

						//distanceX = distanceX * distanceX;
						//distanceY = distanceY * distanceY;
						//distanceZ = distanceZ * distanceZ;

						//distance = distanceX + distanceY + distanceZ;

						//if (minDistance > distance) {
						//	minDistance = distance;
						//	NNID = MappedIndexTemp;
						//}

						//if (distanceY > minDistance) {
						//	//exitFlag = true;
						//	//atomicAdd_block(&NExitedThreadsRight, 1);
						//	//break;
						//}

					}

				}
			}

		}

		__syncthreads();

		RightBound = LeftBound - 1;
		LeftBound = RightBound - BLOCKSIZE + 1;
		if (LeftBound < scid) LeftBound = scid;
	}


	if (ICX <= ecid) {
		Dev_NNearestNeighbor[MapedIndex] = NNID;
	}

}

__global__ void Kernel_MyNeighborListCal_SortXY_multipleBox_noshare(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* SortedIndexY, int* ReverseMap_SortedIndexY, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int MappedJCX;
	int MappedJCY;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int MapedIndex;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int ICX;
	int JCX;
	int ICY;
	int JCY;
	bool XorY;  // true for x, false for y

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	ICX = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	XorY = true;

	if (ICX <= ecid) {

		MapedIndex = SortedIndexX[ICX];

		Pos_X = Dev_ClustersPosXYZ[MapedIndex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIndex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIndex][2];

		ICY = ReverseMap_SortedIndexY[MapedIndex];

		/*Right Hand Searching*/
		for (JCX = ICX + 1; JCX <= ecid; JCX++) {

			MappedJCX = SortedIndexX[JCX];

			distanceX = Dev_ClustersPosXYZ[MappedJCX][0] - Pos_X;
			distanceY = Dev_ClustersPosXYZ[MappedJCX][1] - Pos_Y;
			distanceZ = Dev_ClustersPosXYZ[MappedJCX][2] - Pos_Z;

			distanceX = distanceX * distanceX;
			distanceY = distanceY * distanceY;
			distanceZ = distanceZ * distanceZ;

			distance = distanceX + distanceY + distanceZ;

			if (minDistance > distance) {
				minDistance = distance;
				NNID = MappedJCX;
			}

			if (distanceX > minDistance) {
				XorY = true;
				break;
			}

			//Y restrict
			JCY = ICY + JCX - ICX;
			if (JCY < scid) JCY = scid;
			if (JCY > ecid) JCY = ecid;

			if (JCY != ICY) {

				MappedJCY = SortedIndexY[JCY];

				distanceX = Dev_ClustersPosXYZ[MappedJCY][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJCY][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJCY][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJCY;
				}

				if (distanceY > minDistance) {
					XorY = false;
					break;
				}
			}

		}


		/*Left Hand Searching*/
		if (true == XorY) {

			for (JCX = ICX - 1; JCX >= scid; JCX--) {

				MappedJCX = SortedIndexX[JCX];

				distanceX = Dev_ClustersPosXYZ[MappedJCX][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJCX][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJCX][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJCX;
				}

				if (distanceX > minDistance) {
					break;
				}

			}

		}else{

			for (JCY = ICY - 1; JCY >= scid; JCY--) {

				MappedJCY = SortedIndexY[JCY];

				distanceX = Dev_ClustersPosXYZ[MappedJCY][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[MappedJCY][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[MappedJCY][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = MappedJCY;
				}

				if (distanceY > minDistance) {
					break;
				}

			}

		}

		Dev_NNearestNeighbor[MapedIndex] = NNID;
	}
}





__global__ void Kernel_NormalCalcNeighborList_multipleBox(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	int LeftBound;
	int RightBound;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeJC;
	double distance;
	double minDistance;
	int NNID;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;
	__shared__ double Shared_XYZ[BLOCKSIZE][3];

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox* BlockNumEachBox;

	IC = scid + (cid - bid0* BLOCKSIZE);

	minDistance = 1.E32;

	/*Right Hand Searching*/
	LeftBound = scid;
	RightBound = LeftBound + BLOCKSIZE -1;
	//RightBound = RightBound < NClusters ? RightBound : NClusters;

	if (RightBound > ecid) RightBound = ecid;

	if (IC <= ecid) {
		Pos_X = Dev_ClustersPosXYZ[IC][0];
		Pos_Y = Dev_ClustersPosXYZ[IC][1];
		Pos_Z = Dev_ClustersPosXYZ[IC][2];
	}

	while (LeftBound <= RightBound) {

		if ((LeftBound + tid) <= ecid) {
			Shared_XYZ[tid][0] = Dev_ClustersPosXYZ[LeftBound + tid][0];
			Shared_XYZ[tid][1] = Dev_ClustersPosXYZ[LeftBound + tid][1];
			Shared_XYZ[tid][2] = Dev_ClustersPosXYZ[LeftBound + tid][2];
		}

		__syncthreads();

		if (IC <= ecid) {

			for (int JC = LeftBound; JC <= RightBound; JC++) {
				if (JC != IC) {

					relativeJC = JC - LeftBound;

					distance = (Shared_XYZ[relativeJC][0] - Pos_X)*(Shared_XYZ[relativeJC][0] - Pos_X) +
						(Shared_XYZ[relativeJC][1] - Pos_Y)*(Shared_XYZ[relativeJC][1] - Pos_Y) +
						(Shared_XYZ[relativeJC][2] - Pos_Z)*(Shared_XYZ[relativeJC][2] - Pos_Z);

					if (distance < minDistance) {
						NNID = JC;

						minDistance = distance;
					}

				}

			}

		}

		__syncthreads();

		LeftBound = RightBound + 1;
		RightBound = RightBound + BLOCKSIZE - 1;
		//RightBound = RightBound < NClusters ? RightBound : NClusters;
		if (RightBound > ecid) RightBound = ecid;

	}

	if (IC <= ecid) {
		Dev_NNearestNeighbor[IC] = NNID;
	}

}

__global__ void Kernel_NormalCalcNeighborList_multipleBox_noShared(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor) {
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int cid = bid * BLOCKSIZE + tid;
	double Pos_X;
	double Pos_Y;
	double Pos_Z;
	int relativeJC;
	double distance;
	double minDistance;
	double distanceX;
	double distanceY;
	double distanceZ;
	int NNID;
	int IBox;
	int scid;
	int ecid;
	int bid0;
	int IC;

	IBox = bid / BlockNumEachBox;

	scid = IDStartEnd_Dev[IBox][0];
	ecid = IDStartEnd_Dev[IBox][1];

	bid0 = IBox * BlockNumEachBox;

	IC = scid + (cid - bid0 * BLOCKSIZE);

	minDistance = 1.E32;

	if (IC <= ecid) {

		Pos_X = Dev_ClustersPosXYZ[IC][0];
		Pos_Y = Dev_ClustersPosXYZ[IC][1];
		Pos_Z = Dev_ClustersPosXYZ[IC][2];

		for (int JC = scid; JC <= ecid; JC++) {

			if (IC != JC) {

				distanceX = Dev_ClustersPosXYZ[JC][0] - Pos_X;
				distanceY = Dev_ClustersPosXYZ[JC][1] - Pos_Y;
				distanceZ = Dev_ClustersPosXYZ[JC][2] - Pos_Z;

				distanceX = distanceX * distanceX;
				distanceY = distanceY * distanceY;
				distanceZ = distanceZ * distanceZ;

				distance = distanceX + distanceY + distanceZ;

				if (minDistance > distance) {
					minDistance = distance;
					NNID = JC;
				}

			}
		}

		Dev_NNearestNeighbor[IC] = NNID;
	}

}

void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX,ReverseMap_SortedIndexX);

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortX_multipleBox << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}

void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortX_multipleBox_noshare << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}

void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared_LeftRightCohen(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortX_multipleBox_noshare_LeftRightCohen<< < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);
}

void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared_LeftRightCohen_WithYLimit(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;
	double* ExistedCount_Dev;
	double* ExistedCount_Host;
	double TotalExistedCount;
	double* ExistedYZCount_Dev;
	double* ExistedYZCount_Host;
	double TotalExistedYZCount;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);

	ExistedCount_Host = new double[NClusters];
	ExistedYZCount_Host = new double[NClusters];

	cudaMalloc((void**)&ExistedCount_Dev, NClusters * sizeof(double));
	cudaMalloc((void**)&ExistedYZCount_Dev, NClusters * sizeof(double));

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortX_multipleBox_noshare_LeftRightCohen_WithYLimit << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, Dev_NNearestNeighbor, ExistedCount_Dev, ExistedYZCount_Dev);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);


	cudaMemcpy(ExistedCount_Host, ExistedCount_Dev, NClusters * sizeof(double), cudaMemcpyDeviceToHost);

	TotalExistedCount = 0;
	for (int i = 0; i < NClusters; i++) {
		TotalExistedCount = TotalExistedCount + ExistedCount_Host[i];
	}
	std::cout << "The percent of existed cout: " << TotalExistedCount / NClusters << std::endl;

	cudaMemcpy(ExistedYZCount_Host, ExistedYZCount_Dev, NClusters * sizeof(double), cudaMemcpyDeviceToHost);

	TotalExistedYZCount = 0;
	for (int i = 0; i < NClusters; i++) {
		TotalExistedYZCount = TotalExistedYZCount + ExistedYZCount_Host[i];
	}
	std::cout << "The percent of existed YZ cout: " << TotalExistedYZCount / NClusters << std::endl;

}

void My_NeighborListCal_ArbitrayBitonicSortX_multipleBox_noShared_WithYLimit(int NClusters, 
																			 int NBox, 
																			 int **IDStartEnd_Host, 
																			 int **IDStartEnd_Dev, 
																			 double* ToSortDev_ClustersPosX,
																			 double* ToSortDev_ClustersPosY,
																			 double** Dev_ClustersPosXYZ,
																			 int* SortedIndexX,
																			 int* ReverseMap_SortedIndexX,
																			 int* SortedIndexY,
																			 int* ReverseMap_SortedIndexY,
																			 int* Dev_NNearestNeighbor,
																			 int* Host_NNearestNeighbor,
																			 float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;
	int XJumpStride;
	int TotalNXJumpAllBox;
	int** IDSESeg_Dev;
	int** JumpSegmentYRange_Dev;

	TotalNXJumpAllBox = 0;

	XJumpStride = 2;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosY, SortedIndexY, ReverseMap_SortedIndexY);

	for (int IBox = 0; IBox < NBox; IBox++) {
		TotalNXJumpAllBox = TotalNXJumpAllBox + (IDStartEnd_Host[IBox][1] - IDStartEnd_Host[IBox][0]) / XJumpStride + 1;
	}

	cudaMalloc((void**)&IDSESeg_Dev, NBox * sizeof(int*));

	cudaMalloc((void**)&JumpSegmentYRange_Dev, TotalNXJumpAllBox * sizeof(int*));

	JumpSegmentYRange_multipleBox(NClusters,NBox, IDStartEnd_Host, SortedIndexX, ReverseMap_SortedIndexY, XJumpStride, IDSESeg_Dev, JumpSegmentYRange_Dev);

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortX_multipleBox_noshare_WithYLimit << < blocks, threads >> > (BlockNumEachBox, 
																							 IDStartEnd_Dev, 
																							 Dev_ClustersPosXYZ,
																							 SortedIndexX,
																							 ReverseMap_SortedIndexY,
																							 XJumpStride,
																							 IDSESeg_Dev, 
																							 JumpSegmentYRange_Dev,
																							 Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);
}


void My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double** Dev_ClustersPosXYZ, int* SortedIndexX,int* ReverseMap_SortedIndexX, int* SortedIndexY, int* ReverseMap_SortedIndexY, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;
	int *SortedIndexX_Host;
	int *SortedIndexY_Host;
	int *ReverseMap_SortedIndexX_Host;
	int *ReverseMap_SortedIndexY_Host;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosY, SortedIndexY, ReverseMap_SortedIndexY);



	/*Check*/
	SortedIndexX_Host = new int[NClusters];
	SortedIndexY_Host = new int[NClusters];
	ReverseMap_SortedIndexX_Host = new int[NClusters];
	ReverseMap_SortedIndexY_Host = new int[NClusters];

	cudaMemcpy(SortedIndexX_Host, SortedIndexX, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(SortedIndexY_Host, SortedIndexY, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ReverseMap_SortedIndexX_Host, ReverseMap_SortedIndexX, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(ReverseMap_SortedIndexY_Host, ReverseMap_SortedIndexY,  NClusters * sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < NClusters; i++) {

		if (ReverseMap_SortedIndexX_Host[SortedIndexX_Host[i]] != i) {
			std::cout << "It is no true for index i  X: " << " " << SortedIndexX_Host[i] << " " << ReverseMap_SortedIndexX_Host[SortedIndexX_Host[i]] << " " << i << std::endl;
		}

		if (ReverseMap_SortedIndexY_Host[SortedIndexY_Host[i]] != i) {
			std::cout << "It is no true for index i  Y: " << " " << SortedIndexY_Host[i] << " " << ReverseMap_SortedIndexY_Host[SortedIndexY_Host[i]] << " " << i << std::endl;
		}


		if (SortedIndexY_Host[ReverseMap_SortedIndexY_Host[SortedIndexX_Host[i]]] != SortedIndexX_Host[i]) {
			std::cout << "It is wrong for index i between X and Y" << std::endl;
		}

	}


	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortXY_multipleBox << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, ReverseMap_SortedIndexX, SortedIndexY, ReverseMap_SortedIndexY, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}

void My_NeighborListCal_ArbitrayBitonicSortXY_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double* ToSortDev_ClustersPosY, double** Dev_ClustersPosXYZ, int* SortedIndexX, int* ReverseMap_SortedIndexX, int* SortedIndexY, int* ReverseMap_SortedIndexY, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndexX, ReverseMap_SortedIndexX);
	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosY, SortedIndexY, ReverseMap_SortedIndexY);

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	cudaDeviceSynchronize();

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	Kernel_MyNeighborListCal_SortXY_multipleBox_noshare << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndexX, ReverseMap_SortedIndexX, SortedIndexY, ReverseMap_SortedIndexY, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}



void Common_NeighborListCal_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU) {
	dim3 threads;
	dim3 blocks;
	int NB;
	int BlockNumEachBoxtemp;
	int BlockNumEachBox;

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox*NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	Kernel_NormalCalcNeighborList_multipleBox << < blocks, threads >> > (BlockNumEachBox,IDStartEnd_Dev, Dev_ClustersPosXYZ, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerCommonGPU, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
}

void Common_NeighborListCal_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU) {
	dim3 threads;
	dim3 blocks;
	int NB;
	int BlockNumEachBoxtemp;
	int BlockNumEachBox;

	cudaEvent_t StartEvent;
	cudaEvent_t StopEvent;

	cudaEventCreate(&StartEvent);
	cudaEventCreate(&StopEvent);

	cudaEventRecord(StartEvent, 0);

	BlockNumEachBox = 0;

	for (int i = 0; i < NBox; i++) {
		BlockNumEachBoxtemp = (IDStartEnd_Host[i][1] - IDStartEnd_Host[i][0]) / BLOCKSIZE + 1;

		if (BlockNumEachBox < BlockNumEachBoxtemp) BlockNumEachBox = BlockNumEachBoxtemp;
	}

	NB = BlockNumEachBox * NBox;

	blocks = dim3(NB, 1, 1);
	threads = dim3(BLOCKSIZE, 1, 1);

	Kernel_NormalCalcNeighborList_multipleBox_noShared << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerCommonGPU, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);
}


void Common_NeighborListCal_CPU_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double** Host_ClustersPosXYZ, int* Host_NNearestNeighbor) {

	double minDist;
	double Distance;
	for (int IBox = 0; IBox < NBox; IBox++) {
		for (int i = IDStartEnd_Host[IBox][0]; i <= IDStartEnd_Host[IBox][1]; i++) {

			minDist = 1.E16;

			for (int j = IDStartEnd_Host[IBox][0]; j <= IDStartEnd_Host[IBox][1]; j++) {
				if (i != j) {
					Distance = (Host_ClustersPosXYZ[i][0] - Host_ClustersPosXYZ[j][0])*(Host_ClustersPosXYZ[i][0] - Host_ClustersPosXYZ[j][0]) + \
						(Host_ClustersPosXYZ[i][1] - Host_ClustersPosXYZ[j][1])*(Host_ClustersPosXYZ[i][1] - Host_ClustersPosXYZ[j][1]) + \
						(Host_ClustersPosXYZ[i][2] - Host_ClustersPosXYZ[j][2])*(Host_ClustersPosXYZ[i][2] - Host_ClustersPosXYZ[j][2]);

					if (Distance < minDist) {
						minDist = Distance;
						Host_NNearestNeighbor[i] = j;
					}

				}
			}
		}

	}

}