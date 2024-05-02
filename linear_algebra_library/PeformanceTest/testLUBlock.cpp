#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// this file is used to test the performance to find the best block size for LU factorization

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
    std::ofstream outFile("LU_block_size.txt");
    int n = 1024;
    std::vector<float> mat1 = generateRandomMatrix(n, n);
    Matrix<float> matrix1(n, n, mat1);
    //size : 1024 block_size : 1 serial : 5021 ms
    for (int block_size = 30; block_size <= 300; block_size +=5) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result1 = matrix1.LU_Factorization(block_size, block_size);
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " block_size : " << block_size << " serial : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
    return 0;
}


