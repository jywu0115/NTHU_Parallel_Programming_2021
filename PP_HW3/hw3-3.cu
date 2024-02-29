#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <iostream>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda_runtime.h>

#define B 64
#define BLOCK_SIZE 32
#define NUM_THREAD 2

cpu_set_t cpuset;
const int INF = ((1 << 30) - 1);

void CudaSafeCall(cudaError err) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                cudaGetErrorString(err));
        exit(-1);
    }
}

inline int ceil(int a, int b)
{
	return (a + b - 1) / b;
}

__global__ void BFW_Phase_1(int *D, int num_of_vertices_64, int r){

	int offset_k = r * B;
	__shared__ int pivot_block[B][B];

	/* copy D to share memory */
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_D = (offset_k + threadId_j) * num_of_vertices_64 + offset_k + threadId_i;
			pivot_block[threadId_j][threadId_i] = D[index_D];
		}
	}
	
	#pragma unroll 64
	for (int k = 0; k < B; ++k)
	{
		__syncthreads();
		#pragma unroll 2
		for(int offset_y = 0; offset_y < B; offset_y += 32){
			#pragma unroll 2
			for(int offset_x = 0; offset_x < B; offset_x += 32){
				int i = threadIdx.y + offset_y;
				int j = threadIdx.x + offset_x;
				int temp = pivot_block[i][k] + pivot_block[k][j];
				pivot_block[i][j] = min(pivot_block[i][j], temp);
			}
		}
	}

	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_D = (offset_k + threadId_j) * num_of_vertices_64 + (offset_k + threadId_i);
			D[index_D] = pivot_block[threadId_j][threadId_i];
		}
	}
}

__global__ void BFW_Phase_2_Column(int *D, int num_of_vertices_64, int r, int blockId_start_x, int blockId_start_y){

	__shared__ int pivot_block[B][B];
	__shared__ int self_dependent_block[B][B];
	int offset_k = r * B;
	int blockID_j = (blockId_start_y + blockIdx.y) * B;
	int blockID_i = (blockId_start_x + blockIdx.x) * B;

	/* copy D to share memory */
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_sD = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			self_dependent_block[threadId_j][threadId_i] = D[index_sD];
			int index_pD = (offset_k + threadId_j) * num_of_vertices_64 + offset_k + threadId_i;
			pivot_block[threadId_j][threadId_i] = D[index_pD];
		}
	}

	#pragma unroll 64
	for (int k = 0; k < B; ++k)
	{
		__syncthreads();
		#pragma unroll 2
		for(int offset_y = 0; offset_y < B; offset_y += 32){
			#pragma unroll 2
			for(int offset_x = 0; offset_x < B; offset_x += 32){
				int i = threadIdx.y + offset_y;
				int j = threadIdx.x + offset_x;
				int temp = self_dependent_block[i][k] + pivot_block[k][j];
				self_dependent_block[i][j] = min(self_dependent_block[i][j], temp);
			}
		}	
	}

	/* copy sD result back to D*/
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_D = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			D[index_D] = self_dependent_block[threadId_j][threadId_i];
		}
	}
}

__global__ void BFW_Phase_2_Row(int *D, int num_of_vertices_64, int r, int blockId_start_x, int blockId_start_y){

	__shared__ int pivot_block[B][B];
	__shared__ int self_dependent_block[B][B];
	int offset_k = r * B;
	int blockID_j = (blockId_start_y + blockIdx.y) * B;
	int blockID_i = (blockId_start_x + blockIdx.x) * B;

	/* copy D to share memory */
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_sD = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			self_dependent_block[threadId_j][threadId_i] = D[index_sD];
			int index_pD = (offset_k + threadId_j) * num_of_vertices_64 + offset_k + threadId_i;
			pivot_block[threadId_j][threadId_i] = D[index_pD];
		}
	}

	#pragma unroll 64
	for (int k = 0; k < B; ++k)
	{
		__syncthreads();
		#pragma unroll 2
		for(int offset_y = 0; offset_y < B; offset_y += 32){
			#pragma unroll 2
			for(int offset_x = 0; offset_x < B; offset_x += 32){
				int i = threadIdx.y + offset_y;
				int j = threadIdx.x + offset_x;
				int temp = pivot_block[i][k] + self_dependent_block[k][j];
				self_dependent_block[i][j] = min(self_dependent_block[i][j], temp);
			}
		}	
	}

	/* copy sD result back to D*/
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_D = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			D[index_D] = self_dependent_block[threadId_j][threadId_i];
		}
	}
}

__global__ void BFW_Phase_3(int *D, int num_of_vertices_64, int r, int blockId_start_x, int blockId_start_y){
	
	__shared__ int pivot_block_column[B][B];
	__shared__ int pivot_block_row[B][B];
	__shared__ int target_block[B][B];

	int offset_k = r * B;
	int blockID_j = (blockId_start_y + blockIdx.y) * B;
	int blockID_i = (blockId_start_x + blockIdx.x) * B;

	/* copy D to share memory */
	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){
		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_pcD = (offset_k + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			pivot_block_column[threadId_j][threadId_i] = D[index_pcD];
			int index_prD = (blockID_j + threadId_j) * num_of_vertices_64 + offset_k + threadId_i;
			pivot_block_row[threadId_j][threadId_i] = D[index_prD];
			int index_tD = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			target_block[threadId_j][threadId_i] = D[index_tD]; 
		}
	}

	__syncthreads();

	#pragma unroll 64
	for (int k = 0; k < B; ++k)
	{
		#pragma unroll 2
		for(int offset_y = 0; offset_y < B; offset_y += 32){	
			#pragma unroll 2
			for(int offset_x = 0; offset_x < B; offset_x += 32){
				int i = threadIdx.y + offset_y;
				int j = threadIdx.x + offset_x;
				int temp = pivot_block_row[i][k] + pivot_block_column[k][j];
				target_block[i][j] = min(target_block[i][j], temp);
			}
		}
	}

	#pragma unroll 2
	for(int offset_y = 0; offset_y < B; offset_y += 32){

		#pragma unroll 2
		for(int offset_x = 0; offset_x < B; offset_x += 32){
			int threadId_j = threadIdx.y + offset_y;
			int threadId_i = threadIdx.x + offset_x;
			int index_D = (blockID_j + threadId_j) * num_of_vertices_64 + blockID_i + threadId_i;
			D[index_D] = target_block[threadId_j][threadId_i];
		}
	}
	
}


void BLOCK_FLOYD_WARSHALL(int *dev_D, int num_of_vertices_64)
{

	int Round = num_of_vertices_64 / 64;

	dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 num_blocks;
	CPU_ZERO(&cpuset);
	CPU_SET(0, &cpuset);

	/* Calculation */
	for (int r = 0; r < Round; ++r)
	{
		int rem = Round - r - 1;
		/* Phase 1 */
		cudaSetDevice(0);
		num_blocks.x = 1;
		num_blocks.y = 1;
		BFW_Phase_1<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r);
		/* Phase 2 */
		#pragma omp parallel num_threads(2) shared(dev_D)
		{
			unsigned int  cpu_thread_id = omp_get_thread_num();
			
			if(cpu_thread_id == 1){
				sched_setaffinity(0, sizeof(cpuset), &cpuset);					
				if(r > 0){
					
					num_blocks.x = 1;
					num_blocks.y = r;
					cudaSetDevice(cpu_thread_id);
					
					BFW_Phase_2_Column<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r, 0);
					
					num_blocks.x = r;
					num_blocks.y = 1;
					BFW_Phase_2_Row<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, r);
				}

			}

			if(cpu_thread_id == 0){
				sched_setaffinity(0, sizeof(cpuset), &cpuset);
				if(rem > 0){
					num_blocks.x = 1;
					num_blocks.y = rem;
					cudaSetDevice(cpu_thread_id);
					BFW_Phase_2_Column<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r, r + 1);
					num_blocks.x = rem;
					num_blocks.y = 1;
					BFW_Phase_2_Row<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r + 1, r);			
				}
			}
		}
		cudaDeviceSynchronize();
		/* Phase 3 */
		#pragma omp parallel num_threads(2) shared(dev_D)
		{
			unsigned int  cpu_thread_id = omp_get_thread_num();
			if(cpu_thread_id == 1){
				sched_setaffinity(0, sizeof(cpuset), &cpuset);
				
				if(r > 0){
					cudaSetDevice(cpu_thread_id);
					num_blocks.y = r;
					if(rem > 0){
						num_blocks.x = rem;					
						BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r + 1, 0);
					}
					num_blocks.x = r;
					BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, 0);
				}	
			}

			if(cpu_thread_id == 0){
				sched_setaffinity(0, sizeof(cpuset), &cpuset);
				if(rem > 0){	
					cudaSetDevice(cpu_thread_id);
					num_blocks.y = rem;
					if(r > 0){
						num_blocks.x = r;
						BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, r + 1);
					}
					num_blocks.x = rem;	
					BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r + 1, r + 1);	
				}
			}
		}
		cudaDeviceSynchronize();
	}
}

int main(int argc, char **argv)
{
	assert(argc == 3);
	cpu_set_t cpu_set;
	sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
	int num_of_vertices, num_of_edges;
	omp_set_num_threads(NUM_THREAD);
	
	/* CPU I/O */
	FILE *fin = fopen(argv[1], "rb");
	FILE *fout = fopen(argv[2], "w");
	fread(&num_of_vertices, sizeof(int), 1, fin);
	fread(&num_of_edges, sizeof(int), 1, fin);
	int num_of_vertices_64 = (num_of_vertices / 64  + 1) * 64;

	int *D;
	int src_dst_distance_buffer[3];
	int size_D = num_of_vertices_64 * num_of_vertices_64; // set but no use
	/* GPU init */
	CudaSafeCall(cudaMallocManaged(&D, size_D * sizeof(int)));
	
	std::fill_n(D, size_D, INF);

	for(int i = 0; i < num_of_vertices_64; ++i){
		D[i * num_of_vertices_64 + i] = 0;
	}
	
	for (int i = 0; i < num_of_edges; ++i)
	{
		fread(src_dst_distance_buffer, sizeof(int), 3, fin);
		D[src_dst_distance_buffer[0] * num_of_vertices_64 + src_dst_distance_buffer[1]] = src_dst_distance_buffer[2];
	}
	int device = -1;
	cudaGetDevice(&device);
	CudaSafeCall(cudaMemPrefetchAsync(D, size_D * sizeof(int), device, NULL));
	
	cudaSetDevice(1);
   	cudaDeviceEnablePeerAccess(0, 0);
	
    	cudaSetDevice(0);
    	cudaDeviceEnablePeerAccess(1, 0);

	BLOCK_FLOYD_WARSHALL(D, num_of_vertices_64);

	CudaSafeCall(cudaMemPrefetchAsync(D, size_D * sizeof(int), 0, NULL));
	
	/* CPU I/O */
	for(int i = 0; i < num_of_vertices; ++i){
		fwrite(D + i * num_of_vertices_64, sizeof(int), num_of_vertices, fout);
	}
	fclose(fin);
	fclose(fout);
	CudaSafeCall(cudaFree(D));
	cudaError_t error = cudaGetLastError();
    	if(error != cudaSuccess){
       		printf("CUDA error: %s\n", cudaGetErrorString(error));
        	exit(-1);
    	}
	cudaDeviceReset();
	return 0;
}

	