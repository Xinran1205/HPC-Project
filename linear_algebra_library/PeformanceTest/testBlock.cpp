#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>


// this file is used to test the performance to find the best block size for block multiplication

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
//    std::vector<float> mat1 = generateRandomMatrix(100, 100);
//    std::vector<float> mat2 = generateRandomMatrix(100, 100);
//    Matrix<float> matrix1(100, 100, mat1);
//    Matrix<float> matrix2(100, 100, mat2);
//
//    Matrix<float> result1 = matrix1 * matrix2;
//    Matrix<float> result2 = matrix1.blockMultiplication(matrix2, 10);
//    Matrix<float> result3 = matrix1.multiplyParallel(matrix2);
//    Matrix<float> result4 = matrix1.parallelBlockMultiply(matrix2, 10);
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

    // test performance of block multiplication
    // block size n*n
    std::ofstream outFile("block_size.txt");
    int n = 1024;
    std::vector<float> mat1 = generateRandomMatrix(n, n);
    std::vector<float> mat2 = generateRandomMatrix(n, n);
    Matrix<float> matrix1(n, n, mat1);
    Matrix<float> matrix2(n, n, mat2);
    // test naive multiplication
    auto start = std::chrono::high_resolution_clock::now();
    Matrix<float> result = matrix1*matrix2;
    auto end = std::chrono::high_resolution_clock::now();
    outFile << "Time taken for normal multiplication: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    for (int block_size = 1; block_size <= 1; block_size +=1) {
        start = std::chrono::high_resolution_clock::now();
        Matrix<float> result1 = matrix1.blockMultiplication(matrix2, block_size);
        end = std::chrono::high_resolution_clock::now();
        outFile << "Time taken for block multiplication with block size " << block_size << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
}


