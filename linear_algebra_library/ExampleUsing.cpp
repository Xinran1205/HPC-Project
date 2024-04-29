#include "Matrix.hpp"
#include "SparseMatrixCOO.hpp"
#include "SparseMatrixCSR.hpp"

int main() {

    // Initialize two matrices by using different constructors
    std::vector<std::vector<float>> vec{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<float> matrixA(vec);
    Matrix<float> matrixB({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});

    // example of how to use functions
    Matrix<float> matrixC = matrixA + matrixB;

    Matrix<float> matrixD = matrixA * matrixB;

    Matrix<float> matrixE = matrixA.blockMultiplication(matrixB, 3);

    Matrix<float> matrixF = matrixA.parallelBlockMultiply(matrixB, 3);

    Matrix<float> matrixG = matrixA.LU_Factorization(3);

    Matrix<float> matrixH = matrixA.Parallel_LU_Factorization(3);

    // test sparse matrix COO
    SparseMatrixCOO<int> matrix1 = {
            {0, 0, 3, 0, 0},
            {4, 0, 0, 0, 0},
            {0, 0, 0, 0, 0},
    };
    SparseMatrixCOO<int> matrix2 = {
            {1, 0, 3, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 8, 0},
    };

    SparseMatrixCOO<int> result1 = matrix1 + matrix2;

    // test sparse matrix CSR
    SparseMatrixCSR<int> matrix3({
                                        {1, 0, 0, 0},
                                        {0, 0, 3, 0}
                                });

    SparseMatrixCSR<int> matrix4({
                                         {1, 0, 0, 0},
                                         {0, 2, 0, 0}
                                 });
    SparseMatrixCSR<int> result2 = matrix3 + matrix4;

    result2.printNonZeroElements();
    return 0;
}
