#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <cmath>
#include <assert.h>

#define ARR_SIZE 8000

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

void inner_prod(double local_x[], double local_y[], double* dot, int local_rows, int N, MPI_Comm comm, MPI_Request *request)
{
	double local_dot, sum = 0.0;
        int i;
	
	for (i = 0; i < local_rows; i++)
		sum += local_x[i] * local_y[i];

	MPI_Iallreduce(&sum, (void*)dot, 1, MPI_DOUBLE, MPI_SUM, comm, request);
}

void PBiCGStab(double local_A[], double local_Atrans[], double local_b[], double local_x[], double* norm, int local_rows, int N, MPI_Comm comm)
{
	//double *A_trans;
    double *A_x0;
    double *local_r0;
	double *local_r;
    double *local_s;
    double *local_q;
	double *local_t;
	double *local_z;
	double *local_v;
	double *local_u;
	double *local_f0;
	double *local_vprev;
	double *Ax;
	double *Ax_minus_b;
        
	double sigma_prev = 0, pi = 0, phi = 0, tau = 0, alpha = 1, omega = 1, rho = 1; 
	double sigma, beta, rho_prev, delta, gamma, mu, theta, kappa;
	double dot; 
	int i, j, rank;

	MPI_Request request[7];
	MPI_Status status[7];
	MPI_Comm_rank(comm, &rank);

	//A_trans = (double *)malloc(local_rows * N * sizeof(double));
	A_x0 = (double *)malloc(local_rows * sizeof(double));
	local_r0 = (double *)malloc(local_rows * sizeof(double));
	local_r = (double *)malloc(local_rows * sizeof(double));
    local_s = (double *)malloc(local_rows * sizeof(double));
    local_q = (double *)malloc(local_rows * sizeof(double));
	local_t = (double *)malloc(local_rows * sizeof(double));	
	local_z = (double *)malloc(local_rows * sizeof(double));
	local_v = (double *)malloc(local_rows * sizeof(double));
	local_u = (double *)malloc(local_rows * sizeof(double));	
	local_f0 = (double *)malloc(local_rows * sizeof(double));
	local_vprev = (double *)malloc(local_rows * sizeof(double));

	
	// for (i = 0; i < N; i++) {
	// 	for (j = 0; j < local_rows; j++) {
	// 		A_trans[i * N + j] = A[j * N + i];
	// 	}
	// }

	mat_vec(local_A, local_x, A_x0, local_rows, N, comm);
	for (i = 0; i < local_rows; i++) {
		local_r0[i] = local_b[i] - A_x0[i];
		local_r[i] = local_r0[i];

		local_q[i] = 0;
		local_v[i] = 0;
		local_z[i] = 0;
    }

	mat_vec(local_A, local_r0, local_u, local_rows, N, comm);
	mat_vec(local_Atrans, local_r0, local_f0, local_rows, N, comm);
	// for (i = 0; i < local_rows; i++) {
	// 	local_f0[i] = local_u[i];
    //             //local_w[i] = A_r[i];
   	// }
	
	// norm = inner_prod(local_r, local_r, local_rows, N, comm);
	inner_prod(local_r0, local_u, &sigma, local_rows, N, comm, &request[0]);
	//mat_vec(local_A, local_w, A_w, local_rows, N, comm);

	MPI_Waitall(1, request, status);

	/* main bicg loop */
	for (i = 0; i < N; i++) {
		
		rho_prev = rho;
		rho = phi - (omega * sigma_prev) + (omega * alpha * pi);
		delta = (rho / rho_prev) * alpha;
		beta = delta / omega;
		tau = sigma + beta * tau - (delta * pi);
		alpha = rho / tau;

	/* Axpy */
	for (int rows = 0; rows < local_rows; rows++) {
		local_vprev[rows] = local_v[rows];
		local_v[rows] = local_u[rows] + (beta * local_v[rows]) - (delta * local_q[rows]) ;
	}

	/* Computation */
	mat_vec(local_A, local_v, local_q, local_rows, N, comm);

	/* Axpy */
	for (int rows = 0; rows < local_rows; rows++) {
		local_s[rows] = local_r[rows] - (alpha * local_v[rows]);
		local_t[rows] = local_u[rows] - (alpha * local_q[rows]);
		local_z[rows] = (alpha * local_r[rows]) + (beta * local_z[rows]) - (alpha * beta * omega * local_vprev[rows]);
	}

	/* Reduction */
	inner_prod(local_r0, local_s, &phi, local_rows, N, comm, &request[0]);
	inner_prod(local_r0, local_q, &pi, local_rows, N, comm, &request[1]);
	inner_prod(local_f0, local_s, &gamma, local_rows, N, comm, &request[2]);
	inner_prod(local_f0, local_t, &mu, local_rows, N, comm, &request[3]);
	inner_prod(local_s, local_t, &theta, local_rows, N, comm, &request[4]);
	inner_prod(local_t, local_t, &kappa, local_rows, N, comm, &request[5]);
	inner_prod(local_r, local_r, norm, local_rows, N, comm, &request[6]);

	omega = theta / kappa;
	sigma_prev = sigma;
	sigma = gamma - omega * mu;

	/* Axpy */
	for (int rows = 0; rows < local_rows; rows++) {
		local_r[rows] = local_s[rows] - (omega * local_t[rows]);
		local_x[rows] = local_x[rows] + local_z[rows] + (omega * local_s[rows]);
	}

	/* Computation
			mat_vec(local_A, local_x, Ax, local_rows, N, comm);
			for (int rows = 0; rows < local_rows; rows++) {
					temp_sum += pow((Ax[rows] - local_b[rows]), 2);
			}
			if (sqrt(temp_sum) <= tol)
					break;
	*/
	/* Computation */
	mat_vec(local_A, local_r, local_u, local_rows, N, comm);
	MPI_Waitall(7, request, status);

	if (fabs(*norm) <= 1e-1) {
		if (rank == 0) {
			std::cout << "Iterations = " << i << std::endl;
		}
		break;
	}
}

	free(A_x0);
	free(local_r0);
	free(local_r);
	free(local_s);
	free(local_q);
	free(local_t);
	free(local_z);
	free(local_v);
	free(local_u);
	free(local_f0);
	free(local_vprev);
	// if ()
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
	double* global_A, *local_A, *local_x, *local_b, *local_Atrans;
	double start_time, end_time, norm;
	int N, local_rows, rank, size;
	unsigned long long total_size;
	MPI_Comm comm;
	MPI_Status status;
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

	// if (rank != 0)
	
	if(rank == 0) {
		int index = local_rows;
		global_A = (double *)malloc(sizeof(double) * (N * N));
		local_Atrans = (double *)malloc(sizeof(double) * (N * N));

		srand((unsigned) time(NULL) + rank);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				global_A[i * N + j] = (double) rand() / RAND_MAX;
			}
		}

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				local_Atrans[i * N + j] = global_A[j * N + i];
			}
		}

		for (int i = 1; i < size; i++) {
			// int msg_size = local_rows;
			MPI_Send(global_A + index, local_rows * N, MPI_DOUBLE,
				i, i, comm);
			MPI_Send(local_Atrans + index, local_rows * N, MPI_DOUBLE,
				i, i, comm);
			index += local_rows;
		}
		// global_A = (double *) realloc(global_A, local_rows * N * sizeof(double));
		// assert(global_A[0] != NULL);
		local_A = (double *) realloc(global_A, local_rows * N * sizeof(double));
		local_Atrans = (double *) realloc(local_Atrans, local_rows * N * sizeof(double));
		global_A = NULL;
		assert(local_A != NULL);

	} else {
		global_A = NULL;
		local_A = (double *)malloc(sizeof(double) * (local_rows * N));
		local_Atrans = (double *)malloc(sizeof(double) * (local_rows * N));
		assert(local_Atrans != NULL);
		// local_A = (double *) malloc(local_rows * N * sizeof(double));
		// assert(ob[0] != NULL);

		MPI_Recv(local_A, local_rows * N, MPI_DOUBLE, 0, rank, comm, &status);
		MPI_Recv(local_Atrans, local_rows * N, MPI_DOUBLE, 0, rank, comm, &status);
	}

	// MPI_Bcast();

	//local_A = new double[std::size_t(total_size)];
	local_x = (double *)malloc(local_rows * sizeof(double));
	local_b = (double *)malloc(local_rows * sizeof(double));

	for (int i = 0; i < local_rows; i++) {
		local_x[i] = 0;
		local_b[i] = (double) rand() / RAND_MAX;
	}

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
                	local_x[i] = 0;
        	}
		local_A[0 * N + 0] = 1; local_A[0 * N + 1] = 12; local_A[0 * N + 2] = 8; local_A[0 * N + 3] = 4; local_A[0 * N + 4] = 5; local_A[0 * N + 5] = 6;
		local_A[1 * N + 0] = 11; local_A[1 * N + 1] = 7; local_A[1 * N + 2] = 3; local_A[1 * N + 3] = 9; local_A[1 * N + 4] = 1; local_A[1 * N + 5] = 2;
		local_A[2 * N + 0] = 6; local_A[2 * N + 1] = 7; local_A[2 * N + 2] = 1; local_A[2 * N + 3] = 5; local_A[2 * N + 4] = 9; local_A[2 * N + 5] = 3;

		local_b[0] = 1; local_b[1] = 4; local_b[2] = 3;
	}

	if (rank == 1) {
                for (int i = 0; i < local_rows; i++) {
                        local_x[i] = 0;
                }
                local_A[0 * N + 0] = 8; local_A[0 * N + 1] = 9; local_A[0 * N + 2] = 7; local_A[0 * N + 3] = 11; local_A[0 * N + 4] = 4; local_A[0 * N + 5] = 5;
                local_A[1 * N + 0] = 12; local_A[1 * N + 1] = 10; local_A[1 * N + 2] = 1; local_A[1 * N + 3] = 4; local_A[1 * N + 4] = 2; local_A[1 * N + 5] = 8;
                local_A[2 * N + 0] = 3; local_A[2 * N + 1] = 12; local_A[2 * N + 2] = 6; local_A[2 * N + 3] = 7; local_A[2 * N + 4] = 1; local_A[2 * N + 5] = 2;

		local_b[0] = 8; local_b[1] = 7; local_b[2] = 3;
        }
	*/
	
	// srand((unsigned) time(NULL) + rank);
	// for (int i = 0; i < local_rows; i++) {
	// 	local_x[i] = 0;
	// 	local_b[i] = (double) rand() / RAND_MAX;
	// 	for (int j = 0; j < N; j++) {
	// 		local_A[i * N + j] = (double) rand() / RAND_MAX;
	// 	}
	// }
	
	//print_matrix(local_A, N, local_rows, rank, comm);
	//print_vector(local_x, N, local_rows, rank, comm);
	//print_vector(local_b, N, local_rows, rank, comm);
	/*
	* mat_vec(local_A, local_x, local_y, local_rows, N, comm); 
	* print_vector(local_y, N, local_rows, rank, comm); 
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();

	PBiCGStab(local_A, local_Atrans, local_b, local_x, &norm, local_rows, N, comm);
	
	MPI_Barrier(MPI_COMM_WORLD);	
	end_time = MPI_Wtime();
	
	//print_vector(local_x, N, local_rows, rank, comm);
	
	free(local_A);
	free(local_Atrans);
	free(local_x);
	free(local_b);
	MPI_Finalize();

	if (rank == 0) {
                std::cout << "Runtime of BiCG with " << size << " process: " << end_time - start_time << std::endl;
				std::cout << "Error = " << norm << std::endl;
    }

	return 0;
}
