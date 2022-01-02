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
#include <mpi.h>


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


// Prints out matrix as table, and highlights each block
void printMatrix() {
    for (int i=0 ; i<matrix_size ; i++) {
        printf("\n");
        for (int j=0 ; j<matrix_size ; j++){
            int index = i*matrix_size + j;
            printf("%f, ", matrix[i*matrix_size + j]);
        }
    }
    printf("\n\n");
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
    matrix_size = atoi(argv[1]);
    decimal_precision = atoi(argv[2]);
    decimal_value = pow(0.1, decimal_precision);
    matrix = makeMatrix();


    // Relaxation start
    //----------------------------
    //printf("Hello: rank %d, world: %d\n", rank, world);
    MPI_Bcast(matrix, matrix_size*matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("From rank %d\n", rank);
    printMatrix();
    //----------------------------
    
    
    MPI_Finalize();
    return 0;

}