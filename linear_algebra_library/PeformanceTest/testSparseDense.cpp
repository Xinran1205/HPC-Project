#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

// this file is to test the performance of sparse matrix addition with dense matrix addition


// generate a random sparse matrix with 0.01 sparsity
std::vector<std::vector<float>> generateSparseRandomMatrix(int rows, int cols, float sparsity = 0.01) {
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols, 0.0f));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 100.0f);
    std::uniform_int_distribution<int> row_distrib(0, rows - 1);
    std::uniform_int_distribution<int> col_distrib(0, cols - 1);

    int total_elements = rows * cols;
    int num_nonzeros = static_cast<int>(total_elements * sparsity);

    for (int i = 0; i < num_nonzeros; ++i) {
        int rand_row = row_distrib(gen);
        int rand_col = col_distrib(gen);
        while (mat[rand_row][rand_col] != 0.0f) {
            rand_row = row_distrib(gen);
            rand_col = col_distrib(gen);
        }
        mat[rand_row][rand_col] = distrib(gen);
    }
    return mat;
}

int main() {
    std::ofstream outFile("SparseDense.txt");
    for (int n = 256; n <= 16384; n+= 200) {

        std::vector<std::vector<float>> mat1 = generateSparseRandomMatrix(n, n);
        std::vector<std::vector<float>> mat2 = generateSparseRandomMatrix(n, n);
        SparseMatrixCOO<float> matrixCOO1(mat1);
        SparseMatrixCOO<float> matrixCOO2(mat2);

        SparseMatrixCSR<float> matrixCSR1(mat1);
        SparseMatrixCSR<float> matrixCSR2(mat2);

        Matrix<float> matrixDense1(mat1);
        Matrix<float> matrixDense2(mat2);

        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result = matrixDense1 + matrixDense2;
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " dense : "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        SparseMatrixCOO<float> resultCOO = matrixCOO1 + matrixCOO2;
        end = std::chrono::high_resolution_clock::now();

        outFile << "size : " << n << " COO : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        SparseMatrixCSR<float> resultCSR = matrixCSR1 + matrixCSR2;
        end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " CSR : "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
    return 0;

}



