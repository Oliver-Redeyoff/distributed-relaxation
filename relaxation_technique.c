/**
* Parallel relaxation technique with a distributed memory architecture
* Oliver Redeyoff (ohr22)
*
* How to run:
* > mpicc -Wall -o relaxation relaxation_technique.c -lm
* > mpirun ./relaxation 100 3
* where 100 is the size of the matrix and 3 is the decimal precision
*
**/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>


// A block corresponds to the section of the matrix that a consumer process is to operate on
//
// - start_row: the first row of the matrix that the process will operate on
//
// - end_row: the last row of the matrix that the process will operate on
//
// - input_matrix: where we store the section of the matrix that the process needs to operate on, 
//   it includes two extra rows, the one preceding and following the section as these are needed
//   to compute the averages along the edges of the section
//
// - input_matrix_size: the number of cells in the above matrix
//
// - output_matrix: where we store the result of applying a relaxation on the input_matrix, this 
//   does not include the two extra rows, however it does include an extra cell at the start of 
//   the matrix which is used as a flag which indicates if any of the new values are different to
//   the given precision compared to the values in the input_matrix
//
// - output_matrix_size: the number of cells in the above matrix
//
typedef struct block {
    int start_row;
    int end_row;
    double* input_matrix;
    int input_matrix_size;
    double* output_matrix;
    int output_matrix_size;
} BLOCK;


// Declare functions
double* makeMatrix();
BLOCK* makeBlocks();
void relaxMatrix(BLOCK* block);
double getSuroundingAverage(double* cell);
void printMatrix();


// Global variables
int consumer_count;
int matrix_size;
int decimal_precision;
double decimal_value;


// Create array of doubles of length matrix_size^2
double* makeMatrix() {

    // allocate memory for new matrix of given size
    double* matrix = malloc(matrix_size*matrix_size*sizeof(double));

    // put initial values in matrix
    for (int i=0 ; i<matrix_size ; i++) {
        for (int j=0 ; j<matrix_size ; j++){

            // populate with 1.0 if left or top edge, else with 0.0
            if (i==0 || j==0){
                matrix[i*matrix_size + j] = 1.0;
            } else {
                matrix[i*matrix_size + j] = 0.0;
            }

        }
    }

    return matrix;

}


// Function for workload allocation, creates a block for each consumer process
BLOCK* makeBlocks() {

    // allocate enough space for as many blocks as there are consumer processes
    BLOCK* blocks = malloc(consumer_count*sizeof(BLOCK));

    // the fist and last row are not mutable, so do not allocate them
    int mutable_matrix_size = matrix_size-2;

    // get biggest size that blocks can be while remaining equally sized
    int block_size = floor((double)mutable_matrix_size/(double)consumer_count);

    // get the remainder of rows
    int extra_rows_count = mutable_matrix_size%consumer_count;

    // get the number of blocks which will be sized equally
    int equal_block_count = (mutable_matrix_size-extra_rows_count) / block_size;


    // create as many equally sized blocks as possible
    for(int i=0 ; i<equal_block_count ; i++) {

        BLOCK new_block;

        // specify the start and end row of the block
        new_block.start_row = block_size*i + 1;
        new_block.end_row = block_size*(i+1);

        // allocate memory for the input_matrix
        new_block.input_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1 + 2);
        new_block.input_matrix = (double*)malloc(new_block.input_matrix_size*sizeof(double));
        
        // allocate memory for thte output_matrix
        new_block.output_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1) + 1;
        new_block.output_matrix = (double*)malloc(new_block.output_matrix_size*sizeof(double));

        blocks[i] = new_block;

    }

    // if there are some rows left, we need to create another block which will be slightly bigger
    // than the equally sized ones
    if(extra_rows_count != 0) {

        BLOCK new_block;

        // specify the start and end row of the block
        new_block.start_row = mutable_matrix_size - block_size - extra_rows_count + 1;
        new_block.end_row = mutable_matrix_size;

        // allocate memory for the input_matrix
        new_block.input_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1 + 2);
        new_block.input_matrix = (double*)malloc(new_block.input_matrix_size*sizeof(double));
        
        // allocate memory for thte output_matrix
        new_block.output_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1) + 1;
        new_block.output_matrix = (double*)malloc(new_block.output_matrix_size*sizeof(double));

        blocks[consumer_count-1] = new_block;

    }

    return blocks;

}


// Perform relaxation iteration on a block
void relaxMatrix(BLOCK* block) {

    // we will use old_matrix to calculate new values, which we will put in new_matrix

    // old_matrix contains the row preceding the block and following the block, as these
    // are needed to compute the averages along the top and bottom edge of the block
    double* old_matrix = block->input_matrix;
    // the first cell of output_matrix is used as a flag which indicates if any of the
    // new values have changed to the given precision compared to the values in the old_matrix,
    // so we store the new_matrix values starting at output_matrix+1
    double* new_matrix = block->output_matrix+1;

    // this flag will be set to 1.0 if any of the new values have changed to the given precision 
    // compared to the values in the old_matrix
    int value_changed_flag = 0.0;

    // loop through the old_matrix
    for (int i=0 ; i<block->output_matrix_size-1 ; i++) {
        // only modify non-edge values
        if (i%matrix_size != 0 && (i+1)%matrix_size != 0) {
            // compute new value and check if it differs from the previous value to the given precision
            double new_value = getSuroundingAverage(&old_matrix[matrix_size+i]);
            double diff = new_value - old_matrix[matrix_size+i];
            if (diff > decimal_value) {
                value_changed_flag = 1.0;
            }
            new_matrix[i] = new_value;
        } else {
            new_matrix[i] = old_matrix[matrix_size+i];
        }
    }

    // set the value changed flag in the output_matrix
    block->output_matrix[0] = value_changed_flag;

}


// Returns the average of the four cells surrounding a given cell
double getSuroundingAverage(double* cell) {

    double top_value = *(cell - matrix_size);
    double right_value = *(cell + 1);
    double bottom_value = *(cell + matrix_size);
    double left_value = *(cell - 1);

    return (top_value + right_value + bottom_value + left_value)/4;

}


// Prints out given matrix
void printMatrix(double* matrix, int length) {

    for (int i=0 ; i<length ; i++) {
        if (i%matrix_size == 0) {
            printf("\n");
        }
        printf("%f, ", matrix[i]);
    }
    printf("\n\n");

}


// Main function
int main(int argc, char** argv) {

    // initialise MPI
    MPI_Init(&argc, &argv);
    int rank;
    int world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // initialise global variables
    consumer_count = world-1;
    matrix_size = atoi(argv[1]);
    decimal_precision = atoi(argv[2]);
    decimal_value = pow(0.1, decimal_precision);

    // initialise other variables
    double* matrix = makeMatrix();
    double start_time;
    double end_time;


    // allocate workload to the processes
    BLOCK* blocks = makeBlocks();

    // start timer
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    BLOCK my_block;
    if (rank != 0) {
        my_block = blocks[rank-1];
    }


    // relaxation loop, do this until all the values are unchanged to given precision
    while (1) {

        // process with rank 0 is provider, other processes are consumers
        // this is the logic for the provider
        if (rank == 0) {

            // send appropriate section of the current matrix to each consumer
            for (int i=0 ; i<consumer_count ; i++) {

                double* start_pointer = matrix + matrix_size * (blocks[i].start_row-1);
                MPI_Send(start_pointer, blocks[i].input_matrix_size, MPI_DOUBLE, i+1, 99, MPI_COMM_WORLD);

            }


            // get processed sections of the matrix from each consumer
            int block_changed_flag = 0;
            for (int i=0 ; i<consumer_count ; i++) {

                double* start_pointer = matrix + matrix_size * blocks[i].start_row - 1;
                double start_value = *start_pointer;
                
                MPI_Status stat;
                MPI_Recv(start_pointer, blocks[i].output_matrix_size, MPI_DOUBLE, i+1, 99, MPI_COMM_WORLD, &stat);

                // check the value_changed_flag to see if the new block has differed from the previous iteration's one
                // to the given precision
                int value_changed_flag = (int)*start_pointer;
                if (value_changed_flag == 1) {
                    block_changed_flag = 1;
                }
                *start_pointer = start_value;

            }

            // exit the loop if no block differs to the given precision from the previous iteration
            if (block_changed_flag == 0) {
                break;
            }

        }
        

        // this is the logic for the consumers
        if (rank != 0) {

            // receive assigned block from provider
            MPI_Status stat;
            MPI_Recv(my_block.input_matrix, my_block.input_matrix_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &stat);

            // perform relaxation on assigned block, detect if any value has changed
            relaxMatrix(&my_block);

            // communicate processed block back to provider
            MPI_Send(my_block.output_matrix, my_block.output_matrix_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

        }

    }


    // stop timer
    end_time = MPI_Wtime();

    // output matrix and runtime
    printMatrix(matrix, matrix_size*matrix_size);
    printf("Took %f seconds\n", end_time-start_time);
    
    // terminate MPI processes
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    return 1;

}