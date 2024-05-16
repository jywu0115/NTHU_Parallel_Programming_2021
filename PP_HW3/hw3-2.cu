#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <algorithm>

#define B 64
#define BLOCK_SIZE 32

const int INF = ((1 << 30) - 1);

inline int ceil(int a, int b) {
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
	for (int k = 0; k < B; ++k) {
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
	for (int k = 0; k < B; ++k) {
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
	for (int k = 0; k < B; ++k) {
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
	for (int k = 0; k < B; ++k) {
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

void BLOCK_FLOYD_WARSHALL(int *dev_D, int num_of_vertices_64){
	
	int Round = num_of_vertices_64 / 64;
	dim3 num_threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 num_blocks;
	/* Calculation */
	for (int r = 0; r < Round; ++r){
		int rem = Round - r - 1;

		/* Phase 1 */
		num_blocks.x = 1;
		num_blocks.y = 1;
		BFW_Phase_1<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r);
		
		/* Phase 2 */
		num_blocks.x = 1;
		num_blocks.y = Round;
		BFW_Phase_2_Column<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r, 0);
		
		num_blocks.x = Round;
		num_blocks.y = 1;
		BFW_Phase_2_Row<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, r);
		
		/* Phase 3 */
		if(r > 0){
			num_blocks.x = r;
			num_blocks.y = r;
			BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, 0);			
		}

		if(rem > 0 && r > 0){
			num_blocks.x = r;
			num_blocks.y = rem;
			BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, 0, r + 1);
			
			num_blocks.x = rem;
			num_blocks.y = r;
			BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r + 1, 0);
		}
		if(rem > 0){
			num_blocks.x = rem;
			num_blocks.y = rem;
			BFW_Phase_3<<<num_blocks, num_threads>>>(dev_D, num_of_vertices_64, r, r + 1, r + 1);			
		}
	}
	cudaDeviceSynchronize();
}

int main(int argc, char **argv)
{
	assert(argc == 3);
	int num_of_vertices, num_of_edges;
	
	/* CPU I/O */
	FILE *fin = fopen(argv[1], "rb");
	FILE *fout = fopen(argv[2], "w");
	fread(&num_of_vertices, sizeof(int), 1, fin);
	fread(&num_of_edges, sizeof(int), 1, fin);
	int num_of_vertices_64 = (num_of_vertices / 64  + 1) * 64;

	int *host_D;
	int src_dst_distance_buffer[3];
	cudaMallocHost(&host_D, num_of_vertices_64 * num_of_vertices_64 * sizeof(int));
	std::fill_n(host_D, num_of_vertices_64 * num_of_vertices_64, INF);

	for(int i = 0; i < num_of_vertices_64; ++i)
		host_D[i * num_of_vertices_64 + i] = 0;
	
	
	for (int i = 0; i < num_of_edges; ++i) {
		fread(src_dst_distance_buffer, sizeof(int), 3, fin);
		host_D[src_dst_distance_buffer[0] * num_of_vertices_64 + src_dst_distance_buffer[1]] = src_dst_distance_buffer[2];
	}
	/* GPU init */
	int *dev_D = NULL;
	cudaMalloc((void **)&dev_D, num_of_vertices_64 * num_of_vertices_64 * sizeof(int));
	cudaMemcpy(dev_D, host_D, num_of_vertices_64 * num_of_vertices_64 * sizeof(int), cudaMemcpyHostToDevice);
	
	BLOCK_FLOYD_WARSHALL(dev_D, num_of_vertices_64);

	cudaMemcpy(host_D, dev_D, num_of_vertices_64 * num_of_vertices_64 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_D);
	/* CPU I/O */
	for(int i = 0; i < num_of_vertices; ++i)
		fwrite(host_D + i * num_of_vertices_64, sizeof(int), num_of_vertices, fout);
	
	fclose(fin);
	fclose(fout);
	cudaFreeHost(host_D);
	cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
       	printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

	cudaDeviceReset();
	return 0;
}