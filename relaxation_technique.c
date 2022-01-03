/**
* Parallel relaxation technique with a distributed memory architecture
* Oliver Redeyoff
*
* mpicc -Wall -o hellompi relaxation_technique.c -lm
* mpirun ./hellompi 20 2
*
* Strategy:
*
*
**/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <mpi.h>
#include "relaxation_technique.h"

int consumer_count;
int decimal_precision;
double decimal_value;
int matrix_size;


// Returns array of doubles of length matrix_size^2
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


// Returns thread_count number of blocks which each contain a start_row, an
// end_row. No blocks overlap and they cover all the mutable rows of the
// array
BLOCK* makeBlocks() {

    // allocate enough space for as many blocks as there are processes
    BLOCK* blocks = malloc(consumer_count*sizeof(BLOCK));

    int mutable_matrix_size = matrix_size-2;

    // get biggest size that blocks can be while remaining equally sized
    int block_size = floor((double)mutable_matrix_size/(double)consumer_count);
    // get the remainder of rows
    int extra = mutable_matrix_size%consumer_count;
    // get the number of blocks which will be sized equally
    int equal_block_count = (mutable_matrix_size-extra) / block_size;

    for(int i=0 ; i<equal_block_count ; i++) {
        BLOCK new_block;
        new_block.start_row = block_size*i + 1;
        new_block.end_row = block_size*(i+1);

        new_block.input_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1 + 2);
        new_block.input_matrix = (double*)malloc(new_block.input_matrix_size*sizeof(double));
        
        new_block.output_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1) + 1;
        new_block.output_matrix = (double*)malloc(new_block.output_matrix_size*sizeof(double));

        blocks[i] = new_block;
    }

    // if there are some rows left, we need to create another block which will be slightly bigger
    if(extra != 0) {
        BLOCK new_block;
        new_block.start_row = mutable_matrix_size - block_size - extra + 1;
        new_block.end_row = mutable_matrix_size;

        new_block.input_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1 + 2);
        new_block.input_matrix = (double*)malloc(new_block.input_matrix_size*sizeof(double));
        
        new_block.output_matrix_size = matrix_size * (new_block.end_row-new_block.start_row+1) + 1;
        new_block.output_matrix = (double*)malloc(new_block.output_matrix_size*sizeof(double));

        blocks[consumer_count-1] = new_block;
    }

    return blocks;
}


// Perform relaxation iteration on a matrix
void relaxMatrix(BLOCK* block) {

    double* old_padded_matrix = block->input_matrix;
    double* new_matrix = block->output_matrix + 1;

    int value_changed_flag = 0.0;

    for (int i=0 ; i<block->output_matrix_size-1 ; i++) {
        if (i%matrix_size != 0 && (i+1)%matrix_size != 0) {
            double new_value = getSuroundingAverage(&old_padded_matrix[matrix_size+i]);
            double diff = new_value - old_padded_matrix[matrix_size+i];
            if (diff > decimal_value) {
                value_changed_flag = 1.0;
            }
            new_matrix[i] = new_value;
        } else {
            new_matrix[i] = old_padded_matrix[matrix_size+i];
        }
    }

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


// Prints out matrix as table, and highlights each block
void printMatrix(double* matrix, int length) {
    for (int i=0 ; i<length ; i++) {
        if (i%matrix_size == 0) {
            printf("\n");
            //printf("%d  ", i/matrix_size);
        }
        printf("%f, ", matrix[i]);
    }
    printf("\n\n");
}


// Prints info on the created blocks
void printBlocks(BLOCK* blocks) {
    for (int i=0 ; i<consumer_count ; i++) {
        printf("Block %d:\n", i);
        printf("    Start row : %d\n", blocks[i].start_row);
        printf("    End row : %d\n", blocks[i].end_row);
        printf("\n");
    }
}


// Main function
int main(int argc, char** argv) {

    // Initialise MPI
    MPI_Init(&argc, &argv);
    int rank;
    int world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Initialise global variables
    consumer_count = world-1;
    matrix_size = atoi(argv[1]);
    decimal_precision = atoi(argv[2]);
    decimal_value = pow(0.1, decimal_precision);
    double* matrix = makeMatrix();
    double start_time;
    double end_time;


    // Relaxation start
    //----------------------------
    BLOCK* blocks = makeBlocks();

    if (rank == 0) {
        //printBlocks(blocks);
        start_time = MPI_Wtime();
    }

    BLOCK my_block;
    if (rank != 0) {
        my_block = blocks[rank-1];
    }

    // relaxation loop, do this until the values are unchanged to given precision
    while (1) {

        // process with rank 0 is provider, other processes are consumers
        if (rank == 0) {

            // send appropriate rows to each consumer
            for (int i=0 ; i<consumer_count ; i++) {
                double* start_pointer = matrix + matrix_size * (blocks[i].start_row-1);
                MPI_Send(start_pointer, blocks[i].input_matrix_size, MPI_DOUBLE, i+1, 99, MPI_COMM_WORLD);
            }

            // then get responses from each consumer
            int block_changed_flag = 0;
            for (int i=0 ; i<consumer_count ; i++) {
                double* start_pointer = matrix + matrix_size * blocks[i].start_row - 1;
                double start_value = *start_pointer;
                
                MPI_Status stat;
                MPI_Recv(start_pointer, blocks[i].output_matrix_size, MPI_DOUBLE, i+1, 99, MPI_COMM_WORLD, &stat);

                int value_changed_flag = (int)*start_pointer;
                *start_pointer = start_value;

                if (value_changed_flag == 1) {
                    block_changed_flag = 1;
                }
            }

            if (block_changed_flag == 0) {
                end_time = MPI_Wtime();
                break;
            }

            // usleep(50000);
            // system("clear");
            // printMatrix(matrix, matrix_size*matrix_size);

        } else {

            // receive new block from provider
            MPI_Status stat;
            MPI_Recv(my_block.input_matrix, my_block.input_matrix_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &stat);

            // perform relaxation on assigned block, detect if any value has changed
            relaxMatrix(&my_block);

            // communicate relaxed block back to provider
            MPI_Send(my_block.output_matrix, my_block.output_matrix_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

        }

    }
    //----------------------------

    printMatrix(matrix, matrix_size*matrix_size);
    printf("Took %f to complete\n", end_time-start_time);
    
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    return 1;

}