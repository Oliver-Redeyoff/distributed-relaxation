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
int value_change_flag;
int matrix_size;
double* matrix;


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

        blocks[i] = new_block;
    }

    // if there are some rows left, we need to create another block which will be slightly bigger
    if(extra != 0) {
        BLOCK new_block;
        new_block.start_row = mutable_matrix_size - block_size - extra + 1;
        new_block.end_row = mutable_matrix_size;

        blocks[consumer_count-1] = new_block;
    }

    return blocks;
}


// Prints out matrix as table, and highlights each block
void printMatrix(double* matrix) {
    for (int i=0 ; i<matrix_size ; i++) {
        printf("\n");
        printf("%d  ", i);
        for (int j=0 ; j<matrix_size ; j++){
            int index = i*matrix_size + j;
            printf("%f, ", matrix[i*matrix_size + j]);
        }
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
    matrix = makeMatrix();


    // Relaxation start
    //----------------------------
    BLOCK* blocks = makeBlocks();

    if (rank == 0) {
        printBlocks(blocks);
        //printMatrix();
    }
    //printf("Hello: rank %d, world: %d\n", rank, world);

    int my_padded_block_size;
    double* my_padded_block;
    
    if (rank != 0) {
        my_padded_block_size = matrix_size * (blocks[rank-1].end_row - blocks[rank-1].start_row + 2);
        my_padded_block = (double*)malloc(my_padded_block_size*sizeof(double));
    }

    // relaxation loop, do this until the values are unchanged to given precision
    for (int l=0 ; l<2 ; l++) {

        // process with rank 0 is provider, other processes are consumers
        if (rank == 0) {

            // send appropriate rows to each consumer
            for (int i=0 ; i<consumer_count ; i++) {
                int padded_block_size = matrix_size * (blocks[i].end_row - blocks[i].start_row + 2);
                double* padded_block_start_pointer = matrix + matrix_size * (blocks[i].start_row-1);
                printf("Provider sending block to rank %d\n", i+1);
                MPI_Send(padded_block_start_pointer, padded_block_size, MPI_DOUBLE, i+1, 99, MPI_COMM_WORLD);
            }

            // then get responses from each consumer
            for (int i=0 ; i<consumer_count ; i++) {
                int block_size = matrix_size * (blocks[i].end_row - blocks[i].start_row);
                double* block_start_pointer = matrix + matrix_size * blocks[i].start_row;
                MPI_Status stat;
                MPI_Recv(block_start_pointer, block_size, MPI_DOUBLE, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &stat);
                printf("Provider received response from consumer %d\n", stat.MPI_SOURCE);
            }

        } else {

            // receive new block from provider
            MPI_Status stat;
            MPI_Recv(my_padded_block, my_padded_block_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, &stat);
            printf("Consumer %d received block\n", rank);

            // communicate back to provider
            printf("Consumer %d sending response\n", rank);
            double* block_start = my_padded_block + matrix_size;
            int block_size = my_padded_block_size - matrix_size*2;
            MPI_Send(block_start, block_size, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);

        }

    }
    //MPI_Bcast(matrix, matrix_size*matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //printf("From rank %d\n", rank);
    //printMatrix();
    //----------------------------
    
    
    MPI_Finalize();
    return 0;

}