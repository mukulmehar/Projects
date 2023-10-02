#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <cmath>

#define ARR_SIZE 32000

void mat_vec(double local_A[], double local_x[], double local_y[], int local_rows, int N, MPI_Comm comm)
{
        double* x;
        int local_i, j;

        x = (double *)malloc(N * sizeof(double));
        if (x == NULL) {
                std::cout << "Unable to allocate memory for vector x" << std::endl;
                return;
        }
        MPI_Allgather(local_x, local_rows, MPI_DOUBLE, x, local_rows, MPI_DOUBLE, comm);

        for (local_i = 0; local_i < local_rows; local_i++) {
                local_y[local_i] = 0;
                for (j = 0; j < N; j++) {
                        local_y[local_i] += local_A[local_i * N + j] * x[j];
                }
        }

        free(x);
}

double inner_prod(double local_x[], double local_y[], int local_rows, int N, MPI_Comm comm)
{
	double local_dot, dot, sum = 0.0;
        int i;
	
	for (i = 0; i < local_rows; i++)
		sum += local_x[i] * local_y[i];

	MPI_Allreduce(&sum, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
	return dot;
}

void BiCGStab(double local_A[], double local_b[], double local_x[], double *norm, int local_rows, int N, MPI_Comm comm)
{
        double *A_x0;
        double *local_r0;
	double *local_r;
        double *local_p;
        double *local_s;
        double *local_y;
        double *local_q;
        double alpha, omega, beta, r0_s, r0_r, q_y, y_y, temp_r0_ri, norm_temp;
        int i, rank;

		MPI_Comm_rank(comm, &rank);

        A_x0 = (double *)malloc(local_rows * sizeof(double));
        local_r0 = (double *)malloc(local_rows * sizeof(double));
	local_r = (double *)malloc(local_rows * sizeof(double));
        local_p = (double *)malloc(local_rows * sizeof(double));
        local_s = (double *)malloc(local_rows * sizeof(double));
        local_y = (double *)malloc(local_rows * sizeof(double));
        local_q = (double *)malloc(local_rows * sizeof(double));

        mat_vec(local_A, local_x, A_x0, local_rows, N, comm);
        for (i = 0; i < local_rows; i++) {
                local_r0[i] = local_b[i] - A_x0[i];
		local_r[i] = local_r0[i];
                local_p[i] = local_r0[i];
        }

	r0_r = inner_prod(local_r0, local_r, local_rows, N, comm);
	norm_temp = r0_r;
	*norm = norm_temp;

        for (i = 0; i < N; i++) {
                mat_vec(local_A, local_p, local_s, local_rows, N, comm);

		MPI_Barrier(MPI_COMM_WORLD);
		/* Reduction */
		r0_s = inner_prod(local_r0, local_s, local_rows, N, comm);

		alpha = r0_r / r0_s;
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_q[rows] = local_r[rows] - alpha * local_s[rows];

		mat_vec(local_A, local_q, local_y, local_rows, N, comm);
		
		MPI_Barrier(MPI_COMM_WORLD);
		/* Reduction */
		q_y = inner_prod(local_q, local_y, local_rows, N, comm);
		y_y = inner_prod(local_y, local_y, local_rows, N, comm);
		*norm = norm_temp;
		norm_temp = inner_prod(local_r, local_r, local_rows, N, comm);

		omega = q_y / y_y;
		/* Axpy; x = x + alpha * p + omega * q	*/
                for (int rows = 0; rows < local_rows; rows++)
                        local_x[rows] = local_x[rows] + alpha * local_p[rows] + omega * local_q[rows];
		/* Axpy; r = q - omega * y  */
                for (int rows = 0; rows < local_rows; rows++)
                        local_r[rows] = local_q[rows] - omega * local_y[rows];
		if(fabs(norm_temp) <= 1e-1 || isnan(norm_temp)) {
			if(rank == 0)
				std::cout << "Iterations = " << i << std::endl;
			
			break;
		}		
		temp_r0_ri = r0_r;

		MPI_Barrier(MPI_COMM_WORLD);
		/* Reduction */
		r0_r = inner_prod(local_r0, local_r, local_rows, N, comm);

		beta = (alpha / omega) * (r0_r / temp_r0_ri);
		/* Axpy; p = r + beta(p - omega * s)  */
                for (int rows = 0; rows < local_rows; rows++)
                        local_p[rows] = local_r[rows] + beta * (local_p[rows] - omega * local_s[rows]);

		//MPI_Barrier(MPI_COMM_WORLD);
        }

	free(A_x0);
	free(local_r0);
	free(local_r);
	free(local_p);
	free(local_s);
	free(local_y);
	free(local_q);
	
}

void print_matrix(double local_A[], int N, int local_rows, int rank, MPI_Comm comm)
{
	double *A = NULL;
	int i, j;

	if (rank == 0) {
		A = (double *) malloc(N * N * sizeof(double));
		if (A == NULL)	return;
		MPI_Gather(local_A, local_rows * N, MPI_DOUBLE, A, local_rows * N, MPI_DOUBLE, 0, comm);
		printf("\nThe matrix is:\n");
		for (i = 0; i < N; i++) {
			for (j = 0; j < N; j++) {
				std::cout << A[i * N + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
		free(A);
	} else {
		MPI_Gather(local_A, local_rows * N, MPI_DOUBLE, A, local_rows * N, MPI_DOUBLE, 0, comm);
	}
}

void print_vector(double local_vec[], int N, int local_rows, int rank, MPI_Comm comm)
{
	double* vec = NULL;
	int i;

	if (rank == 0) {
		vec = (double *) malloc(N * sizeof(double));
		if (vec == NULL) {
			std::cout << "Unable to allocate memory for vec in 'print_vector()'" << std::endl;
			return;
		}
		
		MPI_Gather(local_vec, local_rows, MPI_DOUBLE, vec, local_rows, MPI_DOUBLE, 0, comm);
		std::cout << "\nThe vector is:" << std::endl;
		for(i = 0; i < N; i++) {
			std::cout << vec[i] << " ";
		}
		std::cout << std::endl;
		free(vec);
	} else {
		MPI_Gather(local_vec, local_rows, MPI_DOUBLE, vec, local_rows, MPI_DOUBLE, 0, comm);
	}
}

int main()
{
	double* local_A, *local_x, *local_b;
	double start_time, end_time, norm;
	int N, local_rows, rank, size;
	unsigned long long total_size;
	MPI_Comm comm;
	MPI_Init(NULL, NULL);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	if (rank == 0) {
		std::cout << size << std::endl;
	}

	N = ARR_SIZE;
	local_rows = N / size;
	total_size = local_rows * N;

	//local_A = (double *)malloc(sizeof(double) * ((size_t)(local_rows * N)));
	local_A = new double[std::size_t(total_size)];
	local_x = (double *)malloc(local_rows * sizeof(double));
	local_b = (double *)malloc(local_rows * sizeof(double));

	if (local_A == NULL || local_x == NULL || local_b == NULL) {
		if (local_A == NULL) {
			std::cout << "Process " << rank << " unable to allocate memory for local_A." << std::endl;
			perror("main:");
		}
		if (local_x == NULL)
                        std::cout << "main: Process " << rank << " unable to allocate memory for local_x." << std::endl;
		if (local_b == NULL)
                        std::cout << "main: Process " << rank << " unable to allocate memory for local_b." << std::endl;

		std::cout << "Returning from main due to memory allocation error" << std::endl;
		MPI_Finalize();	
		return -1;
	}

	/*
	if (rank == 0) {
		for (int i = 0; i < local_rows; i++) {
                	* local_b[i] = (double) rand() / RAND_MAX; 
                	local_x[i] = 0;
        	}
		local_A[0 * N + 0] = 1; local_A[0 * N + 1] = 12; local_A[0 * N + 2] = 8; local_A[0 * N + 3] = 4; local_A[0 * N + 4] = 5; local_A[0 * N + 5] = 6;
		local_A[1 * N + 0] = 11; local_A[1 * N + 1] = 7; local_A[1 * N + 2] = 3; local_A[1 * N + 3] = 9; local_A[1 * N + 4] = 1; local_A[1 * N + 5] = 2;
		local_A[2 * N + 0] = 6; local_A[2 * N + 1] = 7; local_A[2 * N + 2] = 1; local_A[2 * N + 3] = 5; local_A[2 * N + 4] = 9; local_A[2 * N + 5] = 3;

		local_b[0] = 1; local_b[1] = 4; local_b[2] = 3;
	}

	if (rank == 1) {
                for (int i = 0; i < local_rows; i++) {
                        *local_y[i] = (double) rand() / RAND_MAX; 
                        local_x[i] = 0;
                }
                local_A[0 * N + 0] = 8; local_A[0 * N + 1] = 9; local_A[0 * N + 2] = 7; local_A[0 * N + 3] = 11; local_A[0 * N + 4] = 4; local_A[0 * N + 5] = 5;
                local_A[1 * N + 0] = 12; local_A[1 * N + 1] = 10; local_A[1 * N + 2] = 1; local_A[1 * N + 3] = 4; local_A[1 * N + 4] = 2; local_A[1 * N + 5] = 8;
                local_A[2 * N + 0] = 3; local_A[2 * N + 1] = 12; local_A[2 * N + 2] = 6; local_A[2 * N + 3] = 7; local_A[2 * N + 4] = 1; local_A[2 * N + 5] = 2;

		local_b[0] = 8; local_b[1] = 7; local_b[2] = 3;
        }
	*/

	srand((unsigned) time(NULL) + rank);
	for (int i = 0; i < local_rows; i++) {
		/*local_y[i] = (double) rand() / RAND_MAX;*/
		local_x[i] = 0;
		local_b[i] = (double) rand() / RAND_MAX;
		for (int j = 0; j < N; j++) {
			local_A[i * N + j] = (double) rand() / RAND_MAX;
		}
	}

	/*
	*print_matrix(local_A, N, local_rows, rank, comm);
	*print_vector(local_x, N, local_rows, rank, comm);
	*print_vector(local_b, N, local_rows, rank, comm);
	* mat_vec(local_A, local_x, local_y, local_rows, N, comm); 
	* print_vector(local_y, N, local_rows, rank, comm); 
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();

	BiCGStab(local_A, local_b, local_x, &norm, local_rows, N, comm);
	
	MPI_Barrier(MPI_COMM_WORLD);	
	end_time = MPI_Wtime();
	
	//print_vector(local_x, N, local_rows, rank, comm);
	
	free(local_A);
	free(local_x);
	free(local_b);
	MPI_Finalize();

	if (rank == 0) {
                std::cout << "Runtime of BiCG with " << size << " process: " << end_time - start_time << std::endl;
				std::cout << "Error = " << norm << std::endl;
        }

	return 0;
}
