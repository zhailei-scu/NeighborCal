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

__global__ void Kernel_MyNeighborListCal_multipleBox(int BlockNumEachBox, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor) {
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
	RightBound = LeftBound  + BLOCKSIZE - 1;
	if (RightBound > ecid) RightBound = ecid;

	NRemind = RightBound - LeftBound + 1;

	NExitedThreadsRight = 0;

	if (IC <= ecid) {

		MapedIdex = SortedIndex[IC];

		Pos_X = Dev_ClustersPosXYZ[MapedIdex][0];
		Pos_Y = Dev_ClustersPosXYZ[MapedIdex][1];
		Pos_Z = Dev_ClustersPosXYZ[MapedIdex][2];

	}

	while (LeftBound <= RightBound) {

		if (NExitedThreadsRight >= NRemind) {
			break;
		}

		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndex[LeftBound + tid];

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
	RightBound = LeftBound + BLOCKSIZE - 1;
	if (RightBound > ecid) RightBound = ecid;

	NExitedThreadsLeft = 0;

	while (LeftBound <= RightBound) {

		if (NExitedThreadsLeft >= NRemind) {
			break;
		}


		if ((LeftBound + tid) <= ecid) {

			sortedID = SortedIndex[LeftBound + tid];

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

bool myCompare(double A,double B) {
	return (A < B);
}

bool myCompare2(std::pair<double,int> A, std::pair<double, int> B) {
	return (A.first < B.first);
}


void SimpleSort_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double* ToSortDev_ClustersPosX,int* SortedIndex_Dev) {
	double* ToSortHost_ClustersPosX;
	int* SortedIndex_Host;
	std::vector<std::pair<double, int>> OneBox;

	std::pair<double, int> thePair;

	ToSortHost_ClustersPosX = new double[NClusters];
	SortedIndex_Host = new int[NClusters];

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

			ptr++;
		}
	}

	cudaMemcpy(SortedIndex_Dev,SortedIndex_Host,NClusters * sizeof(int), cudaMemcpyHostToDevice);

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


void My_NeighborListCal_ArbitrayBitonicSort_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod) {
	dim3 threads;
	dim3 blocks;
	int NB;
	cudaError err;
	int noone;
	int BlockNumEachBox;
	int BlockNumEachBoxtemp;

	SimpleSort_multipleBox(NClusters, NBox, IDStartEnd_Host, ToSortDev_ClustersPosX, SortedIndex);

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

	Kernel_MyNeighborListCal_multipleBox << < blocks, threads >> > (BlockNumEachBox, IDStartEnd_Dev, Dev_ClustersPosXYZ, SortedIndex, Dev_NNearestNeighbor);

	cudaDeviceSynchronize();

	cudaEventRecord(StopEvent, 0);

	cudaEventSynchronize(StopEvent);

	cudaEventElapsedTime(&timerMyMethod, StartEvent, StopEvent);

	cudaMemcpy(Host_NNearestNeighbor, Dev_NNearestNeighbor, NClusters * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventDestroy(StartEvent);
	cudaEventDestroy(StopEvent);

}


void Common_NeighborListCal_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU) {
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