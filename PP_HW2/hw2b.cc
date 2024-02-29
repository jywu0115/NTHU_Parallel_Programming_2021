#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <bits/stdc++.h>
#include <emmintrin.h>
#include <nmmintrin.h>

#define CHUNKSIZE 4
int NUM_THREAD;
omp_lock_t mutex;

void Mandelbrot_Set(int *row_buffer, int j, int width, int height, double upper, double lower, double right, double left, long iters)
{

	double t2 = (right - left) / width;
	double y0 = j * ((upper - lower) / height) + lower;
	int align_width = (width >> 1) << 1;
	#pragma omp parallel num_threads(NUM_THREAD)
	{
		__m128d x_128, y_128, temp_128, y0_128, x0_128, length_squared_128, a, b, c, two, four, left_128, t2_128;
		__m128i flag_128, repeats_128, iters_128, gt;
		two = _mm_set_pd1(2);
		four = _mm_set_pd1(4);
		iters_128 = _mm_set1_epi64x(iters - 1);
		y0_128 = _mm_set_pd1(y0);
		t2_128 = _mm_set_pd1(t2);
		left_128 = _mm_set_pd1(left);
		#pragma omp for schedule(dynamic, CHUNKSIZE) nowait
		for (int i = 0; i < align_width; i += 2)
		{
			a = _mm_set_pd(i + 1, i);
			b = _mm_mul_pd(t2_128, a);	  // i * ((right - left) / width)
			x0_128 = _mm_add_pd(left_128, b); // i * ((right - left) / width) + left

			x_128 = _mm_setzero_pd();
			y_128 = _mm_setzero_pd();
			length_squared_128 = _mm_setzero_pd();
			repeats_128 = _mm_setzero_si128();
			flag_128 = _mm_set1_epi64x(1);

			while (flag_128[0] || flag_128[1])
			{
				// temp = x * x - y * y + x0;
				a = _mm_mul_pd(x_128, x_128);	  // x^2
				b = _mm_mul_pd(y_128, y_128);	  // y^2
				c = _mm_sub_pd(a, b);		  // x^2 - y^2
				temp_128 = _mm_add_pd(c, x0_128); // x * x - y * y + x0;

				// y = 2 * x * y + y0;
				a = _mm_mul_pd(x_128, y_128);  // x * y
				b = _mm_mul_pd(a, two);	       // 2 * x * y
				y_128 = _mm_add_pd(b, y0_128); // 2 * x * y + y0

				// x = temp;
				x_128 = temp_128;

				// length_squared = x^2 + y^2;
				a = _mm_mul_pd(temp_128, temp_128);    // x^2
				b = _mm_mul_pd(y_128, y_128);	       // y^2
				length_squared_128 = _mm_add_pd(a, b); // x^2 + y^2;

				repeats_128 = _mm_add_epi64(repeats_128, flag_128);

				a = _mm_cmpge_pd(length_squared_128, four);
				gt = _mm_cmpgt_epi64(repeats_128, iters_128);

				if (a[0] || gt[0])
					flag_128[0] = 0;

				if (a[1] || gt[1])
					flag_128[1] = 0;
			}
			row_buffer[i] = repeats_128[0];
			row_buffer[i + 1] = repeats_128[1];
		}
	}
	if (width != align_width)
	{
		row_buffer[align_width] = row_buffer[align_width - 1];
	}
}

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
	FILE *fp = fopen(filename, "wb");
	assert(fp);
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(png_ptr);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
	png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);
	size_t row_size = 3 * width * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_size);
	for (int y = 0; y < height; ++y)
	{
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x)
		{
			int p = buffer[(height - 1 - y) * width + x];
			png_bytep color = row + x * 3;
			if (p != iters)
			{
				if (p & 16)
				{
					color[0] = 240;
					color[1] = color[2] = p % 16 * 16;
				}
				else
				{
					color[0] = p % 16 * 16;
				}
			}
		}
		png_write_row(png_ptr, row);
	}
	free(row);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
}

void Master_process(int *image, const char *filename, int iters, double left, double right, double lower, double upper, int width, int height, int num_of_process)
{

	int row_count = height - 1, executing_slave_processes = num_of_process - 1;
	int *row_buffer = (int *)malloc(width * sizeof(int));
	assert(row_buffer);
	int *Master_row_buffer = (int *)malloc(width * sizeof(int));
	assert(Master_row_buffer);
	MPI_Status status;
	MPI_Request request;

	#pragma omp parallel num_threads(NUM_THREAD) shared(row_count)
	{
		#pragma omp sections
		{
			#pragma omp section
			{
				int j;
				if (num_of_process != 1)
				{
					for (int i = 1; i < num_of_process; ++i)
					{
						omp_set_lock(&mutex);
						j = row_count;
						--row_count;
						omp_unset_lock(&mutex);
						//printf("Master process send row %d.\n", temp);
						MPI_Isend(&j, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
					}
					while (1)
					{
						MPI_Recv(row_buffer, width, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						omp_set_lock(&mutex);
						j = row_count;
						--row_count;
						omp_unset_lock(&mutex);
						if (j >= 0)
						{
							MPI_Isend(&j, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
						}
						else
						{
							--executing_slave_processes;
						}
						if (status.MPI_TAG != height)
						{
							std::copy(row_buffer, row_buffer + width, image + status.MPI_TAG * width);
						}
						if (executing_slave_processes == 0)
						{
							j = -1;
							for (int i = 1; i < num_of_process; ++i)
							{
								MPI_Isend(&j, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
							}
							break;
						}
					}
				}
			}

			#pragma omp section
			{
				int j;
				while (1)
				{
					omp_set_lock(&mutex);
					j = row_count;
					--row_count;
					omp_unset_lock(&mutex);
					if (j >= 0)
					{
						Mandelbrot_Set(Master_row_buffer, j, width, height, upper, lower, right, left, iters);
						std::copy(Master_row_buffer, Master_row_buffer + width, image + j * width);
					}
					else
					{
						break;
					}
				}
			}
		}
	}
	write_png(filename, iters, width, height, image);
	free(row_buffer);
	free(Master_row_buffer);
}

void Slave_process(int width, int height, double upper, double lower, double left, double right, int iters)
{

	int *Slave_row_buffer = (int *)malloc(width * sizeof(int));
	assert(Slave_row_buffer);
	int j, count = 0;
	double t1 = (upper - lower) / height;
	double t2 = (right - left) / width;
	while (1)
	{
		MPI_Recv(&j, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (j == -1)
		{
			break;
		}
		else if (j < -1)
		{
			MPI_Send(Slave_row_buffer, width, MPI_INT, 0, height, MPI_COMM_WORLD);
			break;
		}
		Mandelbrot_Set(Slave_row_buffer, j, width, height, upper, lower, right, left, iters);
		MPI_Send(Slave_row_buffer, width, MPI_INT, 0, j, MPI_COMM_WORLD);
	}
	free(Slave_row_buffer);
}

int main(int argc, char **argv)
{
	/* detect how many CPUs are available */
	cpu_set_t cpu_set;
	sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

	NUM_THREAD = CPU_COUNT(&cpu_set);
	/* MPI init*/
	int rank_of_process, num_of_process;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_process);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

	/* argument parsing */
	assert(argc == 9);
	const char *filename = argv[1];
	int iters = strtol(argv[2], 0, 10);
	double left = strtod(argv[3], 0);
	double right = strtod(argv[4], 0);
	double lower = strtod(argv[5], 0);
	double upper = strtod(argv[6], 0);
	int width = strtol(argv[7], 0, 10);
	int height = strtol(argv[8], 0, 10);
	int *image;

	if (rank_of_process == 0)
	{
		image = (int *)malloc(width * height * sizeof(int));
		Master_process(image, filename, iters, left, right, lower, upper, width, height, num_of_process);
	}
	else
	{
		Slave_process(width, height, upper, lower, left, right, iters);
	}
	if (rank_of_process == 0)
	{
		free(image);
	}
	MPI_Finalize();
	return 0;
}