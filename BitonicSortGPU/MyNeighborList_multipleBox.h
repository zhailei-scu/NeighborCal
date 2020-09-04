#ifndef _MYNEIGHBORLIST_MULTIPLEBOX_
#define _MYNEIGHBORLIST_MULTIPLEBOX_

extern "C" void Common_NeighborListCal_multipleBox(int NClusters,int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double** Dev_ClustersPosXYZ, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerCommonGPU);

extern "C" void My_NeighborListCal_ArbitrayBitonicSort_multipleBox_noShared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod);

extern "C" void My_NeighborListCal_ArbitrayBitonicSort_multipleBox_Shared(int NClusters, int NBox, int **IDStartEnd_Host, int **IDStartEnd_Dev, double* ToSortDev_ClustersPosX, double** Dev_ClustersPosXYZ, int* SortedIndex, int* Dev_NNearestNeighbor, int* Host_NNearestNeighbor, float &timerMyMethod);


extern "C" void Common_NeighborListCal_CPU_multipleBox(int NClusters, int NBox, int **IDStartEnd_Host, double** Host_ClustersPosXYZ, int* Host_NNearestNeighbor);

#endif
