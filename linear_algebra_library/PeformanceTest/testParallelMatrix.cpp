#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// this file is used to test the performance between serial and parallel version of matrix multiplication

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
    // test performance of block multiplication
    // block size n*n
    std::ofstream outFile("MultiplicationParallel.txt");
    // test naive multiplication

    for (int n = 256; n <= 4096; n+=200) {
        std::vector<float> mat1 = generateRandomMatrix(n, n);
        std::vector<float> mat2 = generateRandomMatrix(n, n);
        Matrix<float> matrix1(n, n, mat1);
        Matrix<float> matrix2(n, n, mat2);

        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result = matrix1.blockMultiplication(matrix2, 100);
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " serial : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        Matrix<float> result1 = matrix1.parallelBlockMultiply(matrix2, 100);
        end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " parallel : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
}
