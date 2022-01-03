typedef struct block {
    int start_row;
    int end_row;
} BLOCK;

double* makeMatrix();
BLOCK* makeBlocks();
void printMatrix();