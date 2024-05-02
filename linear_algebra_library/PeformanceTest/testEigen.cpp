#include "../Library/Matrix.hpp"
#include "../Library/SparseMatrixCOO.hpp"
#include "../Library/SparseMatrixCSR.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Dense>

// this file is used to compare the performance of my LU factorization of the Eigen library

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
// this is for generating the data of the serial version
//    std::ofstream outFile("MyLUvsEigenLU.txt");
//    for (int n=1024; n<=12000; n+=1000){
//        std::vector<float> mat = generateRandomMatrix(n, n);
//        Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(mat.data(), n, n);
//
//        Matrix<float> B(n, n, mat);
//
//        auto start = std::chrono::high_resolution_clock::now();
//        Matrix<float> result = B.LU_Factorization(100,100);
//        auto end = std::chrono::high_resolution_clock::now();
//        outFile << "size : " << n << " LU Factorization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
//
//        start = std::chrono::high_resolution_clock::now();
//        Eigen::PartialPivLU<Eigen::MatrixXf> lu(A);
//        end = std::chrono::high_resolution_clock::now();
//        outFile << "size : " << n << " Eigen LU Factorization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
//    }


// this is for generating the data of the parallel version
    std::ofstream outFile("MyLUvsEigenLUParallel.txt");
    for (int n=1024; n<=12000; n+=1000){
        std::vector<float> mat = generateRandomMatrix(n, n);
        Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(mat.data(), n, n);

        Matrix<float> B(n, n, mat);

        auto start = std::chrono::high_resolution_clock::now();
        Matrix<float> result = B.Parallel_LU_Factorization(100,100);
        auto end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " LU Factorization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        Eigen::PartialPivLU<Eigen::MatrixXf> lu(A);
        end = std::chrono::high_resolution_clock::now();
        outFile << "size : " << n << " Eigen LU Factorization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    }
    return 0;
}

