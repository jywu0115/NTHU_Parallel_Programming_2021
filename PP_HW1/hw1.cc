#include <cstdio>
#include <cstdlib>
#include <bits/stdc++.h>
#include <mpi.h>
#include <math.h>

int Compare(float *send_buffer, float *sub_array, MPI_Offset sub_array_length, int sub_sorted)
{
	if (sub_array[sub_array_length - 1] <= send_buffer[0])
	{
		return 0;
	}
	int merge_length = 2 * sub_array_length;
	float *merge_array = (float *)malloc(merge_length * sizeof(float));
	std::merge(sub_array, sub_array + sub_array_length, send_buffer, send_buffer + sub_array_length, merge_array);
	std::copy(merge_array, merge_array + sub_array_length, sub_array);
	std::copy(merge_array + sub_array_length, merge_array + merge_length, send_buffer);
	free(merge_array);
	return 1;
}

void Single_element_Odd_Even_Sort(float *sub_array, int rank_of_process, int n)
{
	int sub_sorted = 0, global_sorted = 1;
	while (global_sorted != 0)
	{
		global_sorted = 0;
		sub_sorted = 0;
		float send_odd_item, send_even_buffer, send_step1_larger_item, send_step2_larger_item;
		
		// step 1: Even phase
		if (rank_of_process % 2 != 0 && rank_of_process < n)
		{ // odd process
			send_odd_item = sub_array[0];
			MPI_Send(&send_odd_item, 1, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank_of_process % 2 == 0 && rank_of_process != n - 1 && rank_of_process < n)
		{ 
			MPI_Recv(&send_odd_item, 1, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (send_odd_item < sub_array[0])
			{
				send_step1_larger_item = sub_array[0];
				sub_array[0] = send_odd_item;
				sub_sorted = 1;
			}
			else
			{
				send_step1_larger_item = send_odd_item;
			}
			MPI_Send(&send_step1_larger_item, 1, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD);
		}
		if (rank_of_process % 2 != 0 && rank_of_process < n)
		{ // then receive the larger one back
			MPI_Recv(&send_step1_larger_item, 1, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sub_array[0] = send_step1_larger_item;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// step 2: Odd phase
		if (rank_of_process % 2 == 0 && rank_of_process != 0 && rank_of_process < n)
		{ // even process
			send_even_buffer = sub_array[0];
			MPI_Send(&send_even_buffer, 1, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD);
		}
		if (rank_of_process % 2 != 0 && rank_of_process != n - 1 && rank_of_process < n)
		{ // odd process
			MPI_Recv(&send_even_buffer, 1, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (send_even_buffer < sub_array[0])
			{
				send_step2_larger_item = sub_array[0];
				sub_array[0] = send_even_buffer;
				sub_sorted = 1;
			}
			else
			{
				send_step2_larger_item = send_even_buffer;
			}

			MPI_Send(&send_step2_larger_item, 1, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD);
		}
		if (rank_of_process % 2 == 0 && rank_of_process != 0 && rank_of_process < n)
		{ // even process

			MPI_Recv(&send_step2_larger_item, 1, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sub_array[0] = send_step2_larger_item;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Allreduce(&sub_sorted, &global_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}
}

void Parallel_Odd_Even_Sort(float *sub_array, MPI_Offset sub_array_length, MPI_Offset last_sub_array_length, int rank_of_process, int num_of_process, bool has_remainder_element)
{
	int sub_sorted = 0, global_sorted = 1;
	MPI_Request request;
	while (global_sorted != 0)
	{
		sub_sorted = 0;
		global_sorted = 0;
		if (has_remainder_element != true || rank_of_process != num_of_process - 1)
		{
			std::sort(sub_array, sub_array + sub_array_length);
		}
		else
		{
			std::sort(sub_array, sub_array + last_sub_array_length);
		}

		// step 1: Even phase, odd to even

		float *send_buffer = (float *)malloc(sub_array_length * sizeof(float));
		if (rank_of_process % 2 != 0)
		{ // odd process
			std::copy(sub_array, sub_array + sub_array_length, send_buffer);
			MPI_Send(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD);

			MPI_Irecv(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process - 1, 1, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			std::copy(send_buffer, send_buffer + sub_array_length, sub_array);
		}
		else
		{ // even process
			if (rank_of_process != num_of_process - 1)
			{
				MPI_Irecv(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, MPI_STATUS_IGNORE);
				sub_sorted = Compare(send_buffer, sub_array, sub_array_length, sub_sorted);

				MPI_Send(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process + 1, 1, MPI_COMM_WORLD);
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);

		// step 2: Odd phase, even to odd 

		if (rank_of_process % 2 == 0 && rank_of_process != 0)
		{ 
			std::copy(sub_array, sub_array + sub_array_length, send_buffer);
			MPI_Send(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process - 1, 0, MPI_COMM_WORLD);
			MPI_Irecv(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process - 1, 1, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			std::copy(send_buffer, send_buffer + sub_array_length, sub_array);
		}
		else if(rank_of_process % 2 != 0 && rank_of_process != num_of_process - 1)
		{ // rank is odd
			MPI_Irecv(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process + 1, 0, MPI_COMM_WORLD, &request);
			MPI_Wait(&request, MPI_STATUS_IGNORE);
			sub_sorted = Compare(send_buffer, sub_array, sub_array_length, sub_sorted);
			MPI_Send(send_buffer, sub_array_length, MPI_FLOAT, rank_of_process + 1, 1, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		free(send_buffer);
		MPI_Allreduce(&sub_sorted, &global_sorted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char **argv)
{
	if (argc != 4)
	{
		fprintf(stderr, "must provide exactly 3 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	int rank_of_process, num_of_process;
	float *sub_array;
	bool proc_more_than_n = false;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_process);
	MPI_Offset filesize = 0, n = 0, sub_array_length = 0, remainder_sub_array_length = 0, last_sub_array_length = 0;
	MPI_File fin, fout;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fin);			 // open in flie
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fout); // open out file
	MPI_File_get_size(fin, &filesize);
	n = filesize / sizeof(float);
	if (n > num_of_process)
	{
		sub_array_length = floor(n / num_of_process);
		remainder_sub_array_length = n % num_of_process;
	}
	else
	{
		proc_more_than_n = true;
		sub_array_length = 1;
	}
	bool has_remainder_element = (remainder_sub_array_length != 0) ? true : false;
	last_sub_array_length = sub_array_length + remainder_sub_array_length;
	sub_array = (float *)malloc(last_sub_array_length * sizeof(float));
	if (proc_more_than_n == false)
	{
		if (has_remainder_element != true || rank_of_process != num_of_process - 1)
		{
			MPI_File_read_at(fin, sizeof(float) * rank_of_process * sub_array_length, sub_array, sub_array_length, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
		else
		{
			MPI_File_read_at(fin, sizeof(float) * rank_of_process * sub_array_length, sub_array, last_sub_array_length, MPI_FLOAT, MPI_STATUS_IGNORE);
		}

		Parallel_Odd_Even_Sort(sub_array, sub_array_length, last_sub_array_length, rank_of_process, num_of_process, has_remainder_element);

		if (has_remainder_element != true || rank_of_process != num_of_process - 1)
		{
			MPI_File_write_at(fout, sizeof(float) * rank_of_process * sub_array_length, sub_array, sub_array_length, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
		else
		{
			MPI_File_write_at(fout, sizeof(float) * rank_of_process * sub_array_length, sub_array, last_sub_array_length, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
	}
	else
	{ // process more than elements
		if (rank_of_process < n)
		{
			MPI_File_read_at(fin, sizeof(float) * rank_of_process, sub_array, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
		}

		if (n != 1)
		{
			Single_element_Odd_Even_Sort(sub_array, rank_of_process, n);
		}
		if (rank_of_process < n)
		{
			MPI_File_write_at(fout, sizeof(float) * rank_of_process,
					  sub_array, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free(sub_array);
	MPI_File_close(&fin);
	MPI_File_close(&fout);
	MPI_Finalize();
	return 0;
}