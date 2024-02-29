#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <omp.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

#define B 512

int NUM_THREADS;
int sB;
const int INF = ((1 << 30) - 1);
const int V = 6000;
static int D[V][V];

pthread_barrier_t barrier;

typedef struct input_args
{
	int thread_ID;
	int num_vertices;
} input_args;

int ceil(int a, int b)
{
	return (a + b - 1) / b;
}

void print_array(int n)
{
	for (int i = 0; i < n; ++i)
	{
		for(int j = 0; j < n; ++j){
			printf("%d\n ", D[i][j]);
		}
	}
	printf("\n");
}

void Calculation(int num_of_vertices, int Round, int thread_ID,
		 int blockId_start_x, int blockId_start_y, int block_width, int block_height)
{

	int block_end_x = blockId_start_x + block_height;
	int block_end_y = blockId_start_y + block_width;

	for (int b_i = blockId_start_x; b_i < block_end_x; ++b_i)
	{
		for (int b_j = blockId_start_y; b_j < block_end_y; ++b_j)
		{
			int start_k = Round * B;
			int end_k = (Round + 1) * B;
			for (int k = start_k; k < end_k && k < num_of_vertices; ++k)
			{
				int block_internal_start_x = b_i * B + thread_ID * sB;
				int block_internal_end_x = b_i * B + (thread_ID + 1) * sB;
				int block_internal_start_y = b_j * B;
				int block_internal_end_y = (b_j + 1) * B;

				if (block_internal_end_x > num_of_vertices)
				{
					block_internal_end_x = num_of_vertices;
				}
				if (block_internal_end_y > num_of_vertices)
				{
					block_internal_end_y = num_of_vertices;
				}

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i)
				{
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j)
					{
						if (D[i][k] + D[k][j] < D[i][j])
						{
							D[i][j] = D[i][k] + D[k][j];
						}
					}
				}
				pthread_barrier_wait(&barrier);
			}
			
		}
	}
}

void *BLOCK_FLOYD_WARSHALL(void *input)
{

	input_args *args = (input_args *)input;

	cpu_set_t cpuset;
    	CPU_ZERO(&cpuset);
    	CPU_SET(args->thread_ID, &cpuset);
    	pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

	int num_of_vertices = args->num_vertices;
	int thread_ID = args->thread_ID;
	int Round = ceil(num_of_vertices, B);
	for (int r = 0; r < Round; ++r)
	{
		fflush(stdout);
		/* Phase 1, no parallel */
		Calculation(num_of_vertices, r, thread_ID,
			    r, r, 1, 1);
		pthread_barrier_wait(&barrier);

		/* Phase 2*/
		Calculation(num_of_vertices, r, thread_ID,
			    r, 0, r, 1);

		Calculation(num_of_vertices, r, thread_ID,
			    r, r + 1, Round - r - 1, 1);

		Calculation(num_of_vertices, r, thread_ID,
			    0, r, 1, r);

		Calculation(num_of_vertices, r, thread_ID,
			    r + 1, r, 1, Round - r - 1);
		pthread_barrier_wait(&barrier);

		/* Phase 3*/
		Calculation(num_of_vertices, r, thread_ID,
			    0, 0, r, r);

		Calculation(num_of_vertices, r, thread_ID,
			    0, r + 1, Round - r - 1, r);

		Calculation(num_of_vertices, r, thread_ID,
			    r + 1, 0, r, Round - r - 1);

		Calculation(num_of_vertices, r, thread_ID,
			    r + 1, r + 1, Round - r - 1, Round - r - 1);
		pthread_barrier_wait(&barrier);
	}

	pthread_exit(NULL);
}

int main(int argc, char **argv)
{

	cpu_set_t cpu_set;
	sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
	NUM_THREADS = CPU_COUNT(&cpu_set);
	sB = (B % NUM_THREADS == 0) ? B / NUM_THREADS : B / NUM_THREADS + 1;

	assert(argc == 3);

	int num_of_vertices, num_of_edges;

	FILE *fin = fopen(argv[1], "rb");
	FILE *fout = fopen(argv[2], "w");
	fread(&num_of_vertices, sizeof(int), 1, fin);
	fread(&num_of_edges, sizeof(int), 1, fin);

	int src_dst_distance_buffer[3];
	for (int i = 0; i < num_of_vertices; ++i)
	{
		for(int j = 0; j < num_of_vertices; ++j){
			if (i == j)
			{
				D[i][j] = 0;
			}
			else
			{
				D[i][j] = INF;
			}
		}
	}
	for (int i = 0; i < num_of_edges; ++i)
	{
		fread(src_dst_distance_buffer, sizeof(int), 3, fin);
		D[src_dst_distance_buffer[0]][src_dst_distance_buffer[1]] = src_dst_distance_buffer[2];
	}

	pthread_t threads[NUM_THREADS];
	struct input_args input[NUM_THREADS];
	pthread_barrier_init(&barrier, NULL, NUM_THREADS);

	for (int t = 0; t < NUM_THREADS; ++t)
	{
		input[t].thread_ID = t;
		input[t].num_vertices = num_of_vertices;
		pthread_create(&threads[t], NULL, BLOCK_FLOYD_WARSHALL, (void *)&input[t]);
	}
	for (int t = 0; t < NUM_THREADS; ++t)
	{
		pthread_join(threads[t], NULL);
	}

	for (int i = 0; i < num_of_vertices; ++i)
	{
		for(int j = 0; j < num_of_vertices; ++j){
			if (D[i][j] >= INF)
			{
				D[i][j] = INF;
			}
		}
		fwrite(D[i], sizeof(int), num_of_vertices, fout);
	}
	pthread_barrier_destroy(&barrier);
	fclose(fin);
	fclose(fout);
	return 0;
}