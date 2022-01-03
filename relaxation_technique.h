typedef struct block {
    int start_row;
    int end_row;
    int padded_matrix_size;
    double* padded_matrix;
    int mutated_matrix_size;
    double* mutated_matrix;
} BLOCK;

double* makeMatrix();
BLOCK* makeBlocks();

void relaxMatrix(BLOCK* block);
double getSuroundingAverage(double* cell);

void printMatrix();
void printBlocks(BLOCK* blocks);