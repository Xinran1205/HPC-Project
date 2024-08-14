# 高性能线性代数库

## 简介
本项目是一个使用C++17标准库实现的高性能线性代数库，专注于矩阵的LU分解。通过采用分块算法，并使用C++17的execution policy进行并行处理，本库旨在提供一个高效、优化的数学解决方案，支持稠密和稀疏矩阵操作。

## 特性
- **LU分解**：使用分块算法实现并优化了底层数据访问、数据存储以及算法逻辑。
- **并行处理**：利用C++17的execution policy实现算法的并行版本，显著提升性能。
- **矩阵支持**：支持稠密矩阵及稀疏矩阵（CSR和COO格式）。
- **优化算法**：用多种算法为CSR（Compressed Sparse Row）和COO（Compressed Sparse Column）格式实现了加法函数操作。
- **用户友好**：对库进行封装，优化用户使用体验。

## 性能比较
经测试表明，本库的LU分解算法在性能上超过了广泛使用的Eigen库

## 如何使用我的高性能线性代数库包

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
