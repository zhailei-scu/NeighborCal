#ifndef _MYNEIGHBORLIST_
#define _MYNEIGHBORLIST_

extern "C" void Common_NeighborListCal(int NClusters, double** Dev_ClustersPosXYZ,int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU);

extern "C" void My_NeighborListCal_RadixSort_noShared(int NClusters, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor,float &timerMyMethod);

extern "C" void My_NeighborListCal_RadixSort_Shared(int NClusters, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod);

//extern "C" void My_NeighborListCal_ArbitrayBitonicSort(int NClusters, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void Common_NeighborListCal_CPU(int NClusters, double** Host_ClustersPosXYZ, int* Host_NNearestNeighbor);

#endif
