#include <iostream>
#include <vector>
#include <chrono>
#include <random>

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
std::pair<double, double> executeAndMeasure(Func matrixMultiply, const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(mat1, mat2, rows1, cols1, rows2, cols2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    double operations = 2.0 * rows1 * cols2 * cols1;
    double gflops = operations / (seconds * 1e9);
    return {seconds * 1000, gflops};
}

int main() {
    std::vector<int> sizes = {100, 200, 300, 400, 500};
    for (int n : sizes) {
        //default square matrix
        int row1 = n;
        int cols1 = n;
        int row2 = n;
        int cols2 = n;
        auto mat1 = generateRandomMatrix(n, n);
        auto mat2 = generateRandomMatrix(n, n);
        // test
//        int row1 = 2;
//        int cols1 = 3;
//        int row2 = 3;
//        int cols2 = 2;
//
//        std::vector<float> mat1 = {1, 2, 3, 4,5,6};
//        std::vector<float> mat2 = {7, 8, 9, 10, 11, 12};
//        std::vector<float> ans = {58, 64, 139, 154};

        // Test IJK, IKJ, JIK, JKI, KIJ, KJI and output the performance
        std::cout << "IJK: " << executeAndMeasure(IJK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(IJK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;

        std::cout << "IKJ: " << executeAndMeasure(IKJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(IKJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;

        std::cout << "JIK: " << executeAndMeasure(JIK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(JIK_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;

        std::cout << "JKI: " << executeAndMeasure(JKI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(JKI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;

        std::cout << "KIJ: " << executeAndMeasure(KIJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(KIJ_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;

        std::cout << "KJI: " << executeAndMeasure(KJI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).first << " ms, "
        << executeAndMeasure(KJI_matrixMultiply, mat1, mat2,row1,cols1,row2,cols2).second << " GFLOPS" << std::endl;


        std::cout << std::endl;
    }

    return 0;
}