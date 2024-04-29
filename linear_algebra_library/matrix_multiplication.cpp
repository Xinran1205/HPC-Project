#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

//naive matrix multiplication

//ijk
std::vector<float> IJK_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

//ikj
std::vector<float> IKJ_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int i = 0; i < rows1; ++i) {
        for (int k = 0; k < cols1; ++k) {
            for (int j = 0; j < cols2; ++j) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

//jik
std::vector<float> JIK_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int j = 0; j < cols2; ++j) {
        for (int i = 0; i < rows1; ++i) {
            for (int k = 0; k < cols1; ++k) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

//jki
std::vector<float> JKI_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int j = 0; j < cols2; ++j) {
        for (int k = 0; k < cols1; ++k) {
            for (int i = 0; i < rows1; ++i) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

//kij
std::vector<float> KIJ_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int k = 0; k < cols1; ++k) {
        for (int i = 0; i < rows1; ++i) {
            for (int j = 0; j < cols2; ++j) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

//kji
std::vector<float> KJI_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    for (int k = 0; k < cols1; ++k) {
        for (int j = 0; j < cols2; ++j) {
            for (int i = 0; i < rows1; ++i) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    }

    return product;
}

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

template<typename Func>
double executeAndMeasure(Func matrixMultiply, const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(mat1, mat2, rows1, cols1, rows2, cols2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
//    double operations = 2.0 * rows1 * cols2 * cols1;
//    double gflops = operations / (seconds * 1e9);
    return seconds * 1000;
}

int main() {
    std::ofstream outFile("IKJKJI_results.txt");

    std::vector<int> sizes = {256,512,1024,2048,3072};
    for (int n : sizes) {
        int row1 = n;
        int cols1 = n;
        int row2 = n;
        int cols2 = n;
        auto mat1 = generateRandomMatrix(n, n);
        auto mat2 = generateRandomMatrix(n, n);

        double operations = 2.0 * row1 * cols2 * cols1;

        auto time1 = executeAndMeasure(IJK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "IJK: " << time1  << " ms, ";
        outFile << "GFLOPS: " << operations / (time1 * 1e6) << std::endl;

        auto time2 = executeAndMeasure(IKJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "IKJ: " << time2  << " ms, ";
        outFile << "GFLOPS: " << operations / (time2 * 1e6) << std::endl;

        auto time3 = executeAndMeasure(JIK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "JIK: " << time3  << " ms, ";
        outFile << "GFLOPS: " << operations / (time3 * 1e6) << std::endl;

        auto time4 = executeAndMeasure(JKI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "JKI: " << time4  << " ms, ";
        outFile << "GFLOPS: " << operations / (time4 * 1e6) << std::endl;

        auto time5 = executeAndMeasure(KIJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "KIJ: " << time5  << " ms, ";
        outFile << "GFLOPS: " << operations / (time5 * 1e6) << std::endl;

        auto time6 = executeAndMeasure(KJI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2);
        outFile << "KJI: " << time6  << " ms, ";
        outFile << "GFLOPS: " << operations / (time6 * 1e6) << std::endl;

        outFile << std::endl;
    }

    outFile.close();
    return 0;
}