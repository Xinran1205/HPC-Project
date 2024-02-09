#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <mutex>


std::vector<std::vector<float>> IJK_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);
std::vector<std::vector<float>> parallelMatrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2);
std::vector<std::vector<float>> Block_matrixMultiply(const std::vector<std::vector<float>>& mat1, const std::vector<std::vector<float>>& mat2, int blockSize);
// error
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

    std::vector<int> index(rows1);
    std::iota(index.begin(), index.end(), 0);

    // row is from product.begin() to product.end()
    // optimized from & to cols1, cols2, &mat1, &mat2, &product
    std::for_each(std::execution::par, index.begin(), index.end(), [cols1, cols2, &mat1, &mat2, &product](int i) {
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
// not good, looks like each thread is doing all columns
std::vector<std::vector<float>> Parallel_Block_matrixMultiply(const std::vector<std::vector<float>>& mat1,
                                                     const std::vector<std::vector<float>>& mat2, int blockSize) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    // calculate the number of blocks
    int blockCount = (rows1 + blockSize - 1) / blockSize;

    // create a vector of block start indices
    std::vector<int> blockStartIndices(blockCount);
    for (int i = 0; i < blockCount; ++i) {
        blockStartIndices[i] = i * blockSize;
    }

    // Perform block matrix multiplication in parallel for each block
    std::for_each(std::execution::par, blockStartIndices.begin(), blockStartIndices.end(),
                  [rows1, cols1, blockSize, cols2, &mat1, &mat2, &product](int ii) {
        for (int jj = 0; jj < cols2; jj += blockSize) {
            for (int kk = 0; kk < cols1; kk += blockSize) {
                for (int i = ii; i < std::min(ii + blockSize, rows1); ++i) {
                    for (int j = jj; j < std::min(jj + blockSize, cols2); ++j) {
                        for (int k = kk; k < std::min(kk + blockSize, cols1); ++k) {
                            product[i][j] += mat1[i][k] * mat2[k][j];
                        }
                    }
                }
            }
        }
    });

    return product;
}


// one thread for each block
std::vector<std::vector<float>> Parallel_Block_matrixMultiply2(const std::vector<std::vector<float>>& mat1,
                                                              const std::vector<std::vector<float>>& mat2, int blockSize) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<float>> product(rows1, std::vector<float>(cols2, 0.0f));

    // calculate the number of blocks in both dimensions
    int blockRows = (rows1 + blockSize - 1) / blockSize;
    int blockCols = (cols2 + blockSize - 1) / blockSize;

    // Perform block matrix multiplication in parallel for each small block
    // Use nested loops to generate tasks for each block
    std::vector<std::pair<int, int>> blocks;
    for (int i = 0; i < blockRows; ++i) {
        for (int j = 0; j < blockCols; ++j) {
            blocks.emplace_back(i, j);
        }
    }

    std::for_each(std::execution::par, blocks.begin(), blocks.end(),
                  [&](std::pair<int, int> block) {
                      int blockRowStart = block.first * blockSize;
                      int blockColStart = block.second * blockSize;
                      for (int ii = blockRowStart; ii < std::min(blockRowStart + blockSize, rows1); ++ii) {
                          for (int jj = blockColStart; jj < std::min(blockColStart + blockSize, cols2); ++jj) {
                              for (int k = 0; k < cols1; ++k) {
                                  product[ii][jj] += mat1[ii][k] * mat2[k][jj];
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
    std::vector<int> sizes = {30,500,253,14,235,1,2,3};
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
        auto paraBlockProduct = Parallel_Block_matrixMultiply(mat1, mat2, blockSize);
        if (ijkProduct != parallelProduct || ijkProduct != blockProduct || ijkProduct != paraBlockProduct) {
            std::cerr << "Block matrix multiplication is incorrect!" << std::endl;
        }else{
            std::cout << "correct!" << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}
