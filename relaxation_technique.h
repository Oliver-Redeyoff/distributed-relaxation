typedef struct block {
    int start_row;
    int end_row;
    int input_matrix_size;
    double* input_matrix;
    int output_matrix_size;
    double* output_matrix;
} BLOCK;

double* makeMatrix();
BLOCK* makeBlocks();

void relaxMatrix(BLOCK* block);
double getSuroundingAverage(double* cell);

void printMatrix();
void printBlocks(BLOCK* blocks);