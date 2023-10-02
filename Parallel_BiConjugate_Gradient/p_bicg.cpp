#include <iostream>
#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <cmath>
#include <assert.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */


#define ARR_SIZE 8000
#define NO_OF_DATAPOINTS 8000
#define MAX_CHAR_PER_LINE NO_OF_DATAPOINTS * 8 + NO_OF_DATAPOINTS + 100

void print_vector(double local_vec[], int N, int local_rows, int rank, MPI_Comm comm);

void mat_vec(double local_A[], double local_x[], double local_y[], int local_rows, int N, MPI_Comm comm)
{
	int rank;
	rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 1) {
		// std::cout << "Inside mat_vec()" << std::endl;
	}
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
		if(rank == 1) {
		// std::cout << "Leaving Matvec" << std::endl;
	}
}

void inner_prod(double local_x[], double local_y[], double* dot, int local_rows, int N, MPI_Comm comm, MPI_Request *request)
{
	double local_dot, sum = 0.0;
        int i;
	
	for (i = 0; i < local_rows; i++)
		sum += local_x[i] * local_y[i];

	MPI_Iallreduce(&sum, (void*)dot, 1, MPI_DOUBLE, MPI_SUM, comm, request);
}

/* PipeLined BiCG */
void PBiCGStab(double local_A[], double local_b[], double local_x[], double *norm, int local_rows, int N, MPI_Comm comm)
{
        double *A_x0;
	double *A_r;
	double *A_w;
        double *local_r0;
	double *local_r;
        double *local_p;
        double *local_s;
        double *local_y;
        double *local_q;
	double *local_w;
	double *local_t;
	double *local_z;
	double *local_v;
        
	double alpha, omega, beta, r0_s, r0_r, q_y, y_y, temp_r0_ri;
	double r0_w, r0_z;  
	double dot;
    int i;
	
	//MPI_Comm_size(comm, &size);
	MPI_Request request[5];
	MPI_Status status[5];
	int rank;

	MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

        A_x0 = (double *)malloc(local_rows * sizeof(double));
	A_r = (double *)malloc(local_rows * sizeof(double));
	A_w = (double *)malloc(local_rows * sizeof(double));
        local_r0 = (double *)malloc(local_rows * sizeof(double));
	local_r = (double *)malloc(local_rows * sizeof(double));
        local_p = (double *)malloc(local_rows * sizeof(double));
        local_s = (double *)malloc(local_rows * sizeof(double));
        local_y = (double *)malloc(local_rows * sizeof(double));
        local_q = (double *)malloc(local_rows * sizeof(double));
	local_w = (double *)malloc(local_rows * sizeof(double));
	local_t = (double *)malloc(local_rows * sizeof(double));	
	local_z = (double *)malloc(local_rows * sizeof(double));
	local_v = (double *)malloc(local_rows * sizeof(double));	

	mat_vec(local_A, local_x, A_x0, local_rows, N, comm);
	for (i = 0; i < local_rows; i++) {
		local_r0[i] = local_b[i] - A_x0[i];
		local_r[i] = local_r0[i];
		local_p[i] = local_r0[i];
	}

	mat_vec(local_A, local_r0, A_r, local_rows, N, comm);
	for (i = 0; i < local_rows; i++) {
                local_w[i] = A_r[i];
		local_s[i] = local_w[i];
   	}

	mat_vec(local_A, local_w, A_w, local_rows, N, comm);
        for (i = 0; i < local_rows; i++) {
                local_t[i] = A_w[i];
		local_z[i] = local_t[i];
        }

	inner_prod(local_r0, local_r, &r0_r, local_rows, N, comm, &request[0]);
	inner_prod(local_r0, local_w, &r0_w, local_rows, N, comm, &request[1]);
	inner_prod(local_q, local_y, &q_y, local_rows, N, comm, &request[2]);
	
	mat_vec(local_A, local_z, local_v, local_rows, N, comm);

	MPI_Waitall(3, request, status);
	
	// //MPI_Barrier(MPI_COMM_WORLD);
	// if(rank == 1) {
	// 	std::cout << "Before first wait" << std::endl;
	// }
	// MPI_Wait(&request[0], &status[0]);
	// if(rank == 1) {
	// 	std::cout << "After first wait" << std::endl;
	// }
	// MPI_Wait(&request[1], &status[1]);
	// MPI_Wait(&request[1], &status[2]);

	alpha = r0_r / r0_w;
	beta = 0;

	for (i = 0; i < local_rows; i++) {
        local_q[i] = local_r[i] - alpha * local_s[i];
		local_y[i] = local_w[i] - alpha * local_z[i];
    }
	inner_prod(local_y, local_y, &y_y, local_rows, N, comm, &request[0]);
	MPI_Waitall(1,request, status);

	omega = q_y / y_y;
	
	//MPI_Barrier(MPI_COMM_WORLD);

	/* main bicg loop */
	for (i = 0; i < N + 1; i++) {
	
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_p[rows] = local_r[rows] + beta * (local_p[rows] - omega * local_s[rows]);
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_s[rows] = local_w[rows] + beta * (local_s[rows] - omega * local_z[rows]);
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_z[rows] = local_t[rows] + beta * (local_z[rows] - omega * local_v[rows]);
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_q[rows] = local_r[rows] - alpha * local_s[rows];
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_y[rows] = local_w[rows] - alpha * local_z[rows];

		/* Reduction */
		inner_prod(local_q, local_y, &q_y, local_rows, N, comm, &request[0]);
		inner_prod(local_y, local_y, &y_y, local_rows, N, comm, &request[1]);

		/* Computation */
		mat_vec(local_A, local_z, local_v, local_rows, N, comm);
		MPI_Waitall(2, request, status);
		//MPI_Barrier(MPI_COMM_WORLD);

		omega = q_y / y_y;
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_x[rows] = local_x[rows] + alpha * local_p[rows] + omega * local_q[rows];
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_r[rows] = local_q[rows] - omega * local_y[rows];
		/* Axpy */
		for (int rows = 0; rows < local_rows; rows++)
			local_w[rows] = local_y[rows] - omega * (local_t[rows] - alpha * local_v[rows]);
			
		// print_vector(local_x, N, local_rows, rank, comm);
		temp_r0_ri = r0_r;
			
		/* Reduction */
		inner_prod(local_r0, local_r, &r0_r, local_rows, N, comm, &request[0]);
		inner_prod(local_r0, local_w, &r0_w, local_rows, N, comm, &request[1]);
		inner_prod(local_r0, local_s, &r0_s, local_rows, N, comm, &request[2]);
		inner_prod(local_r0, local_z, &r0_z, local_rows, N, comm, &request[3]);
		inner_prod(local_r, local_r, norm, local_rows, N, comm, &request[4]);

		/* Computation */
		mat_vec(local_A, local_w, local_t, local_rows, N, comm);	

		MPI_Waitall(5, request, status);
		//MPI_Barrier(MPI_COMM_WORLD);
		// if (rank == 0)
		// 	std::cout << "Error = " << *norm << std::endl;

		if (fabs(*norm) <= 1e-1) {
			if (rank == 0) {
				std::cout << "Iterations = " << i << std::endl;
			}
			break;
		}
		beta = (alpha / omega) * (r0_r / temp_r0_ri);
		alpha = r0_r / (r0_w + (beta * r0_s) - (beta * omega * r0_z));
	}

	*norm = r0_r;

	free(A_x0);
	free(A_r);
	free(A_w);
	free(local_r0);
	free(local_r);
	free(local_p);
	free(local_s);
	free(local_y);
	free(local_q);
	free(local_w);
	free(local_t);
	free(local_z);
	free(local_v);
	
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

float** file_read(char *filename,      /* input file name */
                  int  *numObjs)       /* no. data objects (local) */
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;
    int count = 0;

    FILE *infile;
    char *line, *ret, *token;
    int   lineLen;

    if ((infile = fopen(filename, "r")) == NULL) {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }

    /* first find the number of objects */
    lineLen = MAX_CHAR_PER_LINE;
    line = (char*) malloc(lineLen);
    assert(line != NULL);

    (*numObjs) = 0;
    while ((fgets(line, lineLen, infile) != NULL) && ((*numObjs) < NO_OF_DATAPOINTS)) {
        /* check each line to find the max line length */
        while (strlen(line) == lineLen-1) {
            /* this line read is not complete */
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);

            /* increase lineLen */
            lineLen += MAX_CHAR_PER_LINE;
            line = (char*) realloc(line, lineLen);
            assert(line != NULL);

            ret = fgets(line, lineLen, infile);
            assert(ret != NULL);
        }

        if (strtok(line, " \t\n") != 0)
            (*numObjs)++;
    }
    rewind(infile);

    rewind(infile);
    printf("File %s numObjs   = %d\n",filename,*numObjs);

    count = 0;

    /* allocate space for objects[][] and read all objects */
    len = (*numObjs) * (*numObjs);
    objects = (float **)malloc((*numObjs) * sizeof(float *));
    assert(objects != NULL);
    objects[0] = (float *)malloc(len * sizeof(float));
    assert(objects[0] != NULL);
    for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numObjs);

    i = 0;
    /* read all objects */
    while (fgets(line, lineLen, infile) != NULL && count < NO_OF_DATAPOINTS) {
        token = strtok(line, ",");
        j = 0;
        while(token != NULL) {
                objects[i][j] = atof(token);
                j++;
                token = strtok(NULL, ",");
        }

        i++;
        count++;
    }

    fclose(infile);
    free(line);

    return objects;
}


/*---< mpi_read() >----------------------------------------------------------*/
float** mpi_read(char     *filename,      /* input file name */
                 int      *numObjs,       /* no. data objects (local) */
                 MPI_Comm  comm)
{
	float    **objects;
	int        i, j, len, divd, rem;
	int        rank, nproc;
	MPI_Status status;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	if (rank == 0) {
		objects = file_read(filename, numObjs);
		if (objects == NULL) *numObjs = -1;
	}

	/* broadcast global numObjs and numCoords to the rest proc */
	MPI_Bcast(numObjs, 1, MPI_INT, 0, comm);

	if (*numObjs == -1) {
		MPI_Finalize();
		exit(1);
	}

	divd = (*numObjs) / nproc;
	rem  = (*numObjs) % nproc;

	if (rank == 0) {
		int index = (rem > 0) ? divd+1 : divd;

		/* index is the numObjs partitioned locally in proc 0 */
		(*numObjs) = index;
		/* distribute objects[] to other processes */
		for (i = 1; i < nproc; i++) {
				int msg_size = (i < rem) ? (divd+1) : divd;
				MPI_Send(objects[index], msg_size * (*numObjs), MPI_FLOAT,
						i, i, comm);
				index += msg_size;
		}

		/* reduce the objects[] to local size */
		objects[0] = (float *) realloc(objects[0],
		(*numObjs)*(*numObjs)*sizeof(float));
			assert(objects[0] != NULL);
			objects    = (float **) realloc(objects, (*numObjs)*sizeof(float*));
			assert(objects != NULL);

	}
	else {
			/*  local numObjs */
			(*numObjs) = (rank < rem) ? divd+1 : divd;

			/* allocate space for data points */
			objects    = (float **) malloc((*numObjs) * sizeof(float *));
			assert(objects != NULL);
			objects[0] = (float*) malloc((*numObjs) * (*numObjs) * sizeof(float));
			assert(objects[0] != NULL);
			for (i=1; i<(*numObjs); i++)
					objects[i] = objects[i-1] + (*numObjs);

			MPI_Recv(objects[0], (*numObjs) * (*numObjs), MPI_FLOAT, 0,
					rank, comm, &status);
	}

	return objects;                        
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

	local_A = (double *)malloc(sizeof(double) * (local_rows * N));
	//local_A = new double[std::size_t(total_size)];
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

	
	// if (rank == 0) {
	// 	for (int i = 0; i < local_rows; i++) { 
    //             	local_x[i] = 0;
    //     	}
	// 	local_A[0 * N + 0] = 1; local_A[0 * N + 1] = 12; local_A[0 * N + 2] = 8; local_A[0 * N + 3] = 4; local_A[0 * N + 4] = 5; local_A[0 * N + 5] = 6;
	// 	local_A[1 * N + 0] = 11; local_A[1 * N + 1] = 7; local_A[1 * N + 2] = 3; local_A[1 * N + 3] = 9; local_A[1 * N + 4] = 1; local_A[1 * N + 5] = 2;
	// 	local_A[2 * N + 0] = 6; local_A[2 * N + 1] = 7; local_A[2 * N + 2] = 1; local_A[2 * N + 3] = 5; local_A[2 * N + 4] = 9; local_A[2 * N + 5] = 3;

	// 	local_b[0] = 1; local_b[1] = 4; local_b[2] = 3;
	// }

	// if (rank == 1) {
	// for (int i = 0; i < local_rows; i++) {
	// 		// *local_y[i] = (double) rand() / RAND_MAX; 
	// 		local_x[i] = 0;
	// }
	// local_A[0 * N + 0] = 8; local_A[0 * N + 1] = 9; local_A[0 * N + 2] = 7; local_A[0 * N + 3] = 11; local_A[0 * N + 4] = 4; local_A[0 * N + 5] = 5;
	// local_A[1 * N + 0] = 12; local_A[1 * N + 1] = 10; local_A[1 * N + 2] = 1; local_A[1 * N + 3] = 4; local_A[1 * N + 4] = 2; local_A[1 * N + 5] = 8;
	// local_A[2 * N + 0] = 3; local_A[2 * N + 1] = 12; local_A[2 * N + 2] = 6; local_A[2 * N + 3] = 7; local_A[2 * N + 4] = 1; local_A[2 * N + 5] = 2;

	// local_b[0] = 8; local_b[1] = 7; local_b[2] = 3;
	// }
	
	
	srand((unsigned) time(NULL) + rank);
	for (int i = 0; i < local_rows; i++) {
		/*local_y[i] = (double) rand() / RAND_MAX;*/
		local_x[i] = 0;
		local_b[i] = (double) rand() / RAND_MAX;
		for (int j = 0; j < N; j++) {
			local_A[i * N + j] = (double) rand() / RAND_MAX;
		}
	}
	

	// print_matrix(local_A, N, local_rows, rank, comm);
	// print_vector(local_x, N, local_rows, rank, comm);
	// print_vector(local_b, N, local_rows, rank, comm);
	/*
	* mat_vec(local_A, local_x, local_y, local_rows, N, comm); 
	* print_vector(local_y, N, local_rows, rank, comm); 
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	start_time = MPI_Wtime();

	PBiCGStab(local_A, local_b, local_x, &norm, local_rows, N, comm);
	
	MPI_Barrier(MPI_COMM_WORLD);	
	end_time = MPI_Wtime();
	
	// print_vector(local_x, N, local_rows, rank, comm);
	
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
