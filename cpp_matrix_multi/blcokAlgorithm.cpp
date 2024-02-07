#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>

// Block matrix multiplication

std::vector<std::vector<float>> IJK_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);
std::vector<std::vector<float>> parallelMatrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);
std::vector<std::vector<float>> Block_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2, int blockSize);
std::vector<std::vector<float>> Parallel_Block_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2, int blockSize);


//ijk naive matrix multiplication
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

// parallel naive matrix multiplication
std::vector<std::vector<float>> parallelMatrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }
    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    //initialize the product matrix with 0
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    std::for_each(std::execution::par, product.begin(), product.end(), [&](auto& row) {
        auto i = &row - &product[0];
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    });

    return product;
}

// Block matrix multiplication (without parallelization)
std::vector<std::vector<float>> Block_matrixMultiply(const std::vector<std::vector<float>>& mat1,
                                                     const std::vector<std::vector<float>>& mat2, int blockSize) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    // Perform block matrix multiplication
    for (int ii = 0; ii < rows1; ii += blockSize) {
        for (int jj = 0; jj < cols2; jj += blockSize) {
            for (int kk = 0; kk < cols1; kk += blockSize) {
                // use min to avoid out of range access
                for (int i = ii; i < std::min(ii + blockSize, rows1); ++i) {
                    for (int j = jj; j < std::min(jj + blockSize, cols2); ++j) {
                        for (int k = kk; k < std::min(kk + blockSize, cols1); ++k) {
                            product[i][j] += mat1[i][k] * mat2[k][j];
                        }
                    }
                }
            }
        }
    }

    return product;
}


// Optimized Block matrix multiplication using C++17 features
// error
std::vector<std::vector<float>> Parallel_Block_matrixMultiply(const std::vector<std::vector<float>>& mat1,
                                                     const std::vector<std::vector<float>>& mat2, int blockSize) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    // Perform block matrix multiplication in parallel for each block
    std::for_each(std::execution::par, product.begin(), product.end(), [&](auto& row) {
        int i = &row - &product[0]; // Compute the current row index
        for (int jj = 0; jj < cols2; jj += blockSize) {
            for (int kk = 0; kk < cols1; kk += blockSize) {
                for (int ii = i; ii < std::min(i + blockSize, rows1); ++ii) {
                    for (int j = jj; j < std::min(jj + blockSize, cols2); ++j) {
                        float sum = 0.0f;
                        for (int k = kk; k < std::min(kk + blockSize, cols1); ++k) {
                            sum += mat1[ii][k] * mat2[k][j];
                        }
                        product[ii][j] += sum;
                    }
                }
            }
        }
    });

    return product;
}


// Execute matrix multiplication and measure time and GFLOPS
template<typename Func>
std::pair<double, double> executeAndMeasure(Func matrixMultiply, const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2, int blockSize = 0) {
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(mat1, mat2, blockSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double seconds = duration.count();
    int n = mat1.size();
    double operations = 2.0 * n * n * n;
    //GFLOPS means Giga Floating Point Operations Per Second
    // divide 1e9 to convert operations to Giga operations
    double gflops = operations / (seconds * 1e9); // GFLOPS
    return {seconds * 1000, gflops}; // Return execution time in milliseconds and GFLOPS
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

int main() {
    int blockSize = 64; // Define a suitable block size based on your system's cache size
    std::vector<int> sizes = {100,13,23,17,2,5};
    for (int n : sizes) {
        auto mat1 = generateRandomMatrix(n);
        auto mat2 = generateRandomMatrix(n);

        // Test block matrix multiplication and output the performance
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        auto [time, gflops] = executeAndMeasure(Block_matrixMultiply, mat1, mat2, blockSize);
        std::cout << "Block Matrix Multiplication: " << time << " ms, " << gflops << " GFLOPS" << std::endl;

        // check if the block matrix multiplication is correct
        auto ijkProduct = IJK_matrixMultiply(mat1, mat2);
        auto parallelProduct = parallelMatrixMultiply(mat1, mat2);
        auto blockProduct = Block_matrixMultiply(mat1, mat2, blockSize);
//        auto blockProduct = Parallel_Block_matrixMultiply(mat1, mat2, blockSize);
        if (ijkProduct != parallelProduct || ijkProduct != blockProduct) {
            std::cerr << "Block matrix multiplication is incorrect!" << std::endl;
        }else{
            std::cout << "correct!" << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
