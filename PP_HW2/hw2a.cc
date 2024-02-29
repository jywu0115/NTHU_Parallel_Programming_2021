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
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <nmmintrin.h>


int pixel_count;
cpu_set_t cpuset;
pthread_mutex_t mutex;

typedef struct Calculate_args
{
	int thread_ID;
	int num_threads;
	int *image;
	int iters;
	double left;
	double right;
	double lower;
	double upper;
	int width;
	int height;
} Calculate_args;

void *Calculate_Mandelbrot_set(void *calc_args)
{
	Calculate_args *args = (Calculate_args *)calc_args;
	CPU_ZERO(&cpuset);
	CPU_SET(args->thread_ID, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

	int pixel_index, i, j, align_width = (args->width >> 1) << 1;
	double t1 = (args->upper - args->lower) / args->height;
	double t2 = (args->right - args->left) / args->width;

	__m128d x_128, y_128, temp_128, y0_128, x0_128, length_squared_128, a, b, c, two, four, left_128, t2_128;
	__m128i flag_128, repeats_128, iters_128, gt;

	two = _mm_set_pd1(2);
	four = _mm_set_pd1(4);
	iters_128 = _mm_set1_epi64x(args->iters - 1);

	while (1)
	{
		pthread_mutex_lock(&mutex);
		pixel_index = pixel_count;
		pixel_count -= 2;
		pthread_mutex_unlock(&mutex);
		if(pixel_index < 0)
		{
			break;
		}
		j = pixel_index / args->width;
		i = pixel_index % args->width;
		/* Boundary condition */
		if(i == args->width - 1)
		{ 
			// pixel (i, j)
			double y0 = j * t1 + args->lower;
			double x0 = i * t2 + args->left;
			int repeats = 0;
			double x = 0;
			double y = 0;
			double length_squared = 0;
			while (repeats < args->iters && length_squared < 4)
			{
				double temp = x * x - y * y + x0;
				y = 2 * x * y + y0;
				x = temp;
				length_squared = x * x + y * y;
				++repeats;
			}
			args->image[pixel_index] = repeats;
			// pixel (0, j + 1)
			i = 0;
			++j;
			y0 = j * t1 + args->lower;
			x0 = i * t2 + args->left;
			repeats = 0;
			x = 0;
			y = 0;
			length_squared = 0;
			while (repeats < args->iters && length_squared < 4)
			{
				double temp = x * x - y * y + x0;
				y = 2 * x * y + y0;
				x = temp;
				length_squared = x * x + y * y;
				++repeats;
			}
			args->image[pixel_index + 1] = repeats;
		}
		else
		{
			double y0 = j * t1 + args->lower;
			int row_offset = j * args->width;

			y0_128 = _mm_set_pd1(y0);
			t2_128 = _mm_set_pd1(t2);
			left_128 = _mm_set_pd1(args->left);

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
				a = _mm_mul_pd(x_128, y_128);  //
				b = _mm_mul_pd(a, two);	       // 2 * x * y
				y_128 = _mm_add_pd(b, y0_128); // 2 * x * y + y0

				// x = temp;
				x_128 = temp_128;

				// length_squared = x^2 + y^2;
				a = _mm_mul_pd(temp_128, temp_128);    // x^2
				b = _mm_mul_pd(y_128, y_128);	       // y^2
				length_squared_128 = _mm_add_pd(a, b); // x^2 + y^2;

				// ++repeats
				repeats_128 = _mm_add_epi64(repeats_128, flag_128);

				// repeats < iters && length_squared < 4
				a = _mm_cmpge_pd(length_squared_128, four);
				gt = _mm_cmpgt_epi64(repeats_128, iters_128);
				if (a[0] || gt[0])
				{
					flag_128[0] = 0;
				}
				if (a[1] || gt[1])
				{
					flag_128[1] = 0;
				}
			}
			args->image[row_offset + i] = repeats_128[0];
			args->image[row_offset + i + 1] = repeats_128[1];
		}
	}
	pthread_exit(NULL);
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

int main(int argc, char **argv)
{
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
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

	/*pthread init */
	int num_threads = CPU_COUNT(&cpuset);
	pthread_t threads[num_threads];
	pthread_mutex_init(&mutex, NULL);

	struct Calculate_args calc_args[num_threads];
	
	/* allocate memory for image */
	pixel_count = height * width - 2;
	int *image = (int *)malloc(width * height * sizeof(int));
	assert(image);

	/* mandelbrot set */
	
	for (int t = 0; t < num_threads; ++t)
	{
		calc_args[t].thread_ID = t;
		calc_args[t].num_threads = num_threads;
		calc_args[t].image = image;
		calc_args[t].iters = iters;
		calc_args[t].left = left;
		calc_args[t].right = right;
		calc_args[t].lower = lower;
		calc_args[t].upper = upper;
		calc_args[t].width = width;
		calc_args[t].height = height;
		pthread_create(&threads[t], NULL, Calculate_Mandelbrot_set, (void *)&calc_args[t]);
	}
	for (int t = 0; t < num_threads; ++t)
	{
		pthread_join(threads[t], NULL);
	}
	pthread_mutex_destroy(&mutex);
	/* draw and cleanup */
	write_png(filename, iters, width, height, image);
	free(image);
	return 0;
}