#include <iostream>
#include <vector>
#include <chrono>
#include <random>

//naive matrix multiplication

//ijk
std::vector<std::vector<float>> IJK_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    //initialize the product matrix with 0
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return product;
}

//ikj
std::vector<std::vector<float>> IKJ_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int i = 0; i < rows1; ++i) {
        for (int k = 0; k < cols1; ++k) {
            for (int j = 0; j < cols2; ++j) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return product;
}

//jik
std::vector<std::vector<float>> JIK_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int j = 0; j < cols2; ++j) {
        for (int i = 0; i < rows1; ++i) {
            for (int k = 0; k < cols1; ++k) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return product;
}

//jki
std::vector<std::vector<float>> JKI_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int j = 0; j < cols2; ++j) {
        for (int k = 0; k < cols1; ++k) {
            for (int i = 0; i < rows1; ++i) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return product;
}

//kij
std::vector<std::vector<float>> KIJ_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int k = 0; k < cols1; ++k) {
        for (int i = 0; i < rows1; ++i) {
            for (int j = 0; j < cols2; ++j) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return product;
}

//kji
std::vector<std::vector<float>> KJI_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    for (int k = 0; k < cols1; ++k) {
        for (int j = 0; j < cols2; ++j) {
            for (int i = 0; i < rows1; ++i) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return product;
}

// Generate a random matrix with floating-point numbers
std::vector<std::vector<float>> generateRandomMatrix(int n) {
    std::vector<std::vector<float>> mat(n, std::vector<float>(n));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 100.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = distrib(gen);
        }
    }
    return mat;
}

// Execute matrix multiplication and measure time and GFLOPS
template<typename Func>
std::pair<double, double> executeAndMeasure(Func matrixMultiply, const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(mat1, mat2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    int n = mat1.size();
    double operations = 2.0 * n * n * n;
    //GFOPS means Giga Floating Point Operations Per Second
    // divide 1e9 to convert operations to Giga operations
    double gflops = operations / (seconds * 1e9); // GFLOPS
    return {seconds * 1000, gflops}; // Return execution time in milliseconds and GFLOPS
}

int main() {
    std::vector<int> sizes = {100, 200, 300, 400, 500};
    for (int n : sizes) {
        auto mat1 = generateRandomMatrix(n);
        auto mat2 = generateRandomMatrix(n);

        // Test IJK, IKJ, JIK, JKI, KIJ, KJI and output the performance
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        std::cout << "IJK: " << executeAndMeasure(IJK_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(IJK_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << "IKJ: " << executeAndMeasure(IKJ_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(IKJ_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << "JIK: " << executeAndMeasure(JIK_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(JIK_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << "JKI: " << executeAndMeasure(JKI_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(JKI_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << "KIJ: " << executeAndMeasure(KIJ_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(KIJ_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << "KJI: " << executeAndMeasure(KJI_matrixMultiply, mat1, mat2).first << " ms, "
        << executeAndMeasure(KJI_matrixMultiply, mat1, mat2).second << " GFLOPS" << std::endl;

        std::cout << std::endl;
    }

    return 0;
}