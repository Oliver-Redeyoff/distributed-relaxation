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
//   the matrix which is used as a flag which indicates if any of the new values have changed to
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


double* makeMatrix();
BLOCK* makeBlocks();

void relaxMatrix(BLOCK* block);
double getSuroundingAverage(double* cell);

void printMatrix();
void printBlocks(BLOCK* blocks);