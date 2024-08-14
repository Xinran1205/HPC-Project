# High-Performance Linear Algebra Library
- zh_CN [简体中文](/README.zh_CN.md)
## Introduction
This project is a high-performance linear algebra library implemented using the C++17 standard library, focusing on matrix LU Factorization. By using block algorithms and leveraging C++17's execution policy for parallel processing, this library aims to provide an efficient and optimized mathematical solution, supporting both dense and sparse matrix operations.

## Features
- **LU Factorization**: Utilizes block algorithms to optimize underlying data access, storage, and algorithm logic.
- **Parallel Processing**: Implements parallel versions of algorithms using C++17's execution policy, significantly enhancing performance.
- **Matrix Support**: Supports dense matrices and sparse matrices (CSR and COO formats).
- **Optimized Algorithms**: Implements multiple function operations for CSR (Compressed Sparse Row) and COO (Compressed Sparse Column) formats.
- **User-Friendly**: The library is encapsulated to optimize the user experience.

## Performance Comparison
The library's LU decomposition algorithm surpasses the widely-used Eigen library in performance.

## Library Usage Examples

This section provides a detailed example demonstrating how to utilize the library for basic matrix operations using dense and sparse matrices. The following C++ code snippet shows the usage of various matrix functionalities including matrix addition, multiplication, LU factorization, and operations on sparse matrices.

```cpp
#include "Matrix.hpp"
#include "SparseMatrixCOO.hpp"
#include "SparseMatrixCSR.hpp"

int main() {
    // Initialize two matrices with different constructors
    std::vector<std::vector<float>> vec{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<float> matrixA(vec);
    Matrix<float> matrixB({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}});

    // Example of how to use functions
    Matrix<float> matrixC = matrixA + matrixB; // Matrix addition
    Matrix<float> matrixD = matrixA * matrixB; // Matrix multiplication

    // Block matrix multiplication
    Matrix<float> matrixE = matrixA.blockMultiplication(matrixB, 3);
    Matrix<float> matrixF = matrixA.parallelBlockMultiply(matrixB, 3);

    // LU Factorization
    Matrix<float> matrixG = matrixA.LU_Factorization(3);
    Matrix<float> matrixH = matrixA.Parallel_LU_Factorization(3);

    // Test sparse matrix COO
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

    // Test sparse matrix CSR
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