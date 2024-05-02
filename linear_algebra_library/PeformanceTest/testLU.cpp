#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// this file is used to compare the performance of the serial and parallel version of my LU factorization

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
    std::ofstream outFile("LUParallel.txt");
    // test naive multiplication

    for (int n = 1024; n <= 16384; n*= 2) {
        std::vector<float> mat = generateRandomMatrix(n, n);
        Matrix<float> matrix123(n, n, mat);

        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result = matrix123.LU_Factorization(100,100);
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " serial : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        Matrix<float> result1 = matrix123.Parallel_LU_Factorization(100,100);
        end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " parallel : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
}
