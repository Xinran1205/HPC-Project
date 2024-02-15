#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>

// receive a matrix and its dimension, and perform LU decomposition in place
bool LU_Decomposition(std::vector<float>& mat, int n);
bool BlockLU_Decomposition(std::vector<float>& mat, int n, int blockSize);
bool parallelLU_Decomposition(std::vector<float>& mat, int n);

// receive a matrix and its dimension, and return the L and U matrices
bool LU_Decomposition2(const std::vector<float>& A, std::vector<float>& L, std::vector<float>& U, int n);


//correct method 1
bool LU_Decomposition(std::vector<float>& mat, int n) {
    const float smallVal = 1e-12; // define a small value to check if a number is close to zero

    for (int k = 0; k < n; ++k) {
        if (std::fabs(mat[k*n + k]) < smallVal) {
            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
            return false;
        }
        for (int i = k + 1; i < n; ++i) {
            mat[i*n + k] /= mat[k*n + k];
            for (int j = k + 1; j < n; ++j) {
                mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
            }
        }
    }
    return true;
}

// correct
bool parallelLU_Decomposition(std::vector<float>& mat, int n) {
    const float smallVal = 1e-12;

    for (int k = 0; k < n; ++k) {
        if (std::fabs(mat[k*n + k]) < smallVal) {
            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
            return false;
        }

        std::vector<int> index(n - (k + 1));
        std::iota(index.begin(), index.end(), k + 1);

        std::for_each(std::execution::par, index.begin(), index.end(), [&](int i) {
            mat[i*n + k] /= mat[k*n + k];
            for (int j = k + 1; j < n; ++j) {
                mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
            }
        });
    }
    return true;
}

// correct method 2
bool LU_Decomposition2(const std::vector<float>& A, std::vector<float>& L, std::vector<float>& U, int n) {
    const float EPS = 1e-12;
    L.resize(n * n, 0);
    U.resize(n * n, 0);

    for (int i = 0; i < n; ++i) {
        L[i*n + i] = 1; // initialize the diagonal of L with 1
    }

    for (int k = 0; k < n; ++k) {
        if (std::fabs(A[k*n + k]) < EPS) {
            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
            return false;
        }
        for (int i = k; i < n; ++i) { // calculate U's row
            float sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += L[k*n + j] * U[j*n + i];
            }
            U[k*n + i] = A[k*n + i] - sum;
        }
        for (int i = k + 1; i < n; ++i) { // calculate L's column
            float sum = 0;
            for (int j = 0; j < k; ++j) {
                sum += L[i*n + j] * U[j*n + k];
            }
            L[i*n + k] = (A[i*n + k] - sum) / U[k*n + k];
        }
    }
    return true;
}

// wrong
bool BlockLU_Decomposition(std::vector<float>& mat, int n, int blockSize) {
    const float smallVal = 1e-12;

    for (int kk = 0; kk < n; kk += blockSize) {
        int blockSizeK = std::min(blockSize, n - kk);

        for (int k = kk; k < kk + blockSizeK; ++k) {
            if (std::fabs(mat[k*n + k]) < smallVal) {
                std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
                return false;
            }
            for (int i = k + 1; i < n; ++i) {
                mat[i*n + k] /= mat[k*n + k];
                for (int j = k + 1; j < kk + blockSizeK; ++j) {
                    mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
                }
            }
        }

        // update the rest of the matrix
        for (int ii = kk + blockSizeK; ii < n; ii += blockSize) {
            int blockSizeI = std::min(blockSize, n - ii);
            for (int jj = kk + blockSizeK; jj < n; jj += blockSize) {
                int blockSizeJ = std::min(blockSize, n - jj);

                for (int i = ii; i < ii + blockSizeI; ++i) {
                    for (int j = jj; j < jj + blockSizeJ; ++j) {
                        for (int k = kk; k < kk + blockSizeK; ++k) {
                            mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
                        }
                    }
                }
            }
        }
    }

    return true;
}


int main() {
    int n = 4; // dimension of the matrix
    std::vector<float> A = {2, 1, 1,6,
                            4,-6, 0,8,
                            -2, 7, 2,9,
                            4, 8, 3, 1 }; // matrix to be decomposed

    // method 1
    std::vector<float> mat = A;
    if (!LU_Decomposition(mat, n)) {
        std::cerr << "LU Decomposition failed with the first method." << std::endl;
        return 1;
    }

    // print the result
    std::cout << "Result of LU Decomposition (in-place):" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(8) << mat[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    // parallel method1
    std::vector<float> mat2 = A;
    if(!parallelLU_Decomposition(mat2, n)){
        std::cerr << "Block LU Decomposition failed with the first method." << std::endl;
        return 1;
    }
    if (mat == mat2) {
        std::cout << "Parallel LU Decomposition is correct." << std::endl;
    } else {
        std::cout << "Parallel LU Decomposition is incorrect." << std::endl;
    }



    // method 2
    std::vector<float> L, U;
    if (!LU_Decomposition2(A, L, U, n)) {
        std::cerr << "LU Decomposition failed with the second method." << std::endl;
        return 1;
    }

    // print the result
    std::cout << "L Matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(8) << L[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "U Matrix:" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(8) << U[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}