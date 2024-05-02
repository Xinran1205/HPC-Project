#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>

//naive matrix multiplication
// with test performance of different order of loops

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