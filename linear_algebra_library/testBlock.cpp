#include "Matrix.hpp"
#include "SparseMatrixCOO.hpp"
#include "SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

std::vector<float> generateRandomMatrix(int rows, int cols) {
    std::vector<float> mat(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 100.0f);

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = distrib(gen);
    }
    return mat;
}

int main() {
    // test correctness of block multiplication
//    std::vector<int> mat1 = generateRandomMatrix(100, 100);
//    std::vector<int> mat2 = generateRandomMatrix(100, 100);
//    Matrix<int> matrix1(100, 100, mat1);
//    Matrix<int> matrix2(100, 100, mat2);
//
//    Matrix<int> result1 = matrix1 * matrix2;
//    Matrix<int> result2 = matrix1.blockMultiplication(matrix2, 10);
//    Matrix<int> result3 = matrix1.multiplyParallel(matrix2);
//    Matrix<int> result4 = matrix1.parallelBlockMultiply(matrix2, 10);
//
//    for (int i = 0; i < 100; ++i) {
//        for (int j = 0; j < 100; ++j) {
//            if (result1(i, j) != result4(i, j) || result3!= result4 || result2(i, j) != result4(i, j)) {
//                std::cout << "Incorrect result" << std::endl;
//                return 1;
//            }
//        }
//    }
//    std::cout << "Correct result" << std::endl;

    // 测试不同块大小的矩阵乘法
    // 矩阵大小为2048 * 2048
    std::ofstream outFile("block_size.txt");
    std::vector<float> mat1 = generateRandomMatrix(2048, 2048);
    std::vector<float> mat2 = generateRandomMatrix(2048, 2048);
    Matrix<float> matrix1(2048, 2048, mat1);
    Matrix<float> matrix2(2048, 2048, mat2);
    // 测试普通矩阵乘法
    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> result = matrix1 * matrix2;
    auto end = std::chrono::high_resolution_clock::now();
    outFile << "Time taken for normal multiplication: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    for (int block_size = 2; block_size <= 2048; block_size *= 2) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result = matrix1.blockMultiplication(matrix2, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "Time taken for block multiplication with block size " << block_size << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
}


