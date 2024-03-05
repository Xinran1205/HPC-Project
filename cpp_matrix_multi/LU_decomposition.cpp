#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>

// receive a matrix and its dimension, and perform LU decomposition in place
bool DGETF2(std::vector<float>& mat, int n,int startRow, int startCol, int blockSize);
bool BlockLU_Decomposition(std::vector<float>& mat, int n, int blockSize);
bool parallelLU_Decomposition(std::vector<float>& mat, int n);
int IDAMAX(const std::vector<float>& mat, int n, int k, int startRow, int endRow);
void DSWAP(std::vector<float>& mat, int n, int row1, int row2, int startCol, int endCol);
void DSCAL(std::vector<float>& mat, int n, int col, int startCol, int endCol);
//void DGER(std::vector<float>& mat, int n, int k, int startRow, int endRow, int startCol, int endCol);

// DGEMM
std::vector<float> Parallel_Block_matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2, int blockSize) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

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
                  [blockSize,rows1,cols2,cols1,&mat1,&mat2,&product](std::pair<int, int> block) {
                      int blockRowStart = block.first * blockSize;
                      int blockColStart = block.second * blockSize;
                      for (int ii = blockRowStart; ii < std::min(blockRowStart + blockSize, rows1); ++ii) {
                          for (int jj = blockColStart; jj < std::min(blockColStart + blockSize, cols2); ++jj) {
                              for (int k = 0; k < cols1; ++k) {
                                  product[ii * cols2 + jj] += mat1[ii * cols1 + k] * mat2[k * cols2 + jj];
                              }
                          }
                      }
                  });

    return product;
}

//IDAMAX
// find the index of the maximum element in a column
int IDAMAX(const std::vector<float>& mat, int n, int k, int startRow, int endRow) {
    int pivot = k;
    float maxVal = std::fabs(mat[startRow*n + k]);

    for (int i = startRow + 1; i < endRow; ++i) {
        float val = std::fabs(mat[i*n + k]);
        if (val > maxVal) {
            maxVal = val;
            pivot = i;
        }
    }

    return pivot;
}

// DSWAP
// swap two rows of a matrix but only within a given range of columns
void DSWAP(std::vector<float>& mat, int n, int row1, int row2, int startCol, int endCol) {
    if (row1 == row2) return;

    for (int i = startCol; i < endCol; ++i) {
        std::swap(mat[row1*n + i], mat[row2*n + i]);
    }
}

// DSCAL
// scale a column of a matrix but only within a given range of rows
void DSCAL(std::vector<float>& mat, int n, int k, int startRow, int endRow) {
    for (int i = startRow+1; i < endRow; ++i) {
        mat[i*n + k] /= mat[k*n + k];
    }
}

// DGER
// perform a rank-k update on a matrix but only within a given range of rows and columns
void DGER(std::vector<float>& mat, int n, int k, int startRow, int endRow, int startCol, int endCol) {
    for (int i = startRow+1; i < endRow; ++i) {
        for (int j = startCol+1; j < endCol; ++j) {
            mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
        }
    }
}

// correct!
bool DGETF2(std::vector<float>& mat, int n, int startRow, int startCol, int blockSize) {
    const float smallVal = 1e-12; // define a small value to check if a number is close to zero

    int endRow = std::min(startRow + blockSize, n);
    int endCol = std::min(startCol + blockSize, n);
    //endRow = 4, endCol = 4
    for (int k = startRow; k < endRow; ++k) {
        int pivot = IDAMAX(mat, n, k, k, endRow);
        if (std::fabs(mat[pivot*n + k]) < smallVal) {
            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
            return false;
        }
        DSWAP(mat, n, k, pivot, 0, endCol);
        DSCAL(mat, n, k, k, endRow);
        DGER(mat, n, k,k, endRow, k, endCol);
    }
    return true;
}

bool DGETRF(std::vector<float>& mat, int n, int blockSize) {
    const float smallVal = 1e-12;

    for (int k = 0; k < n; k += blockSize) {
        if (!DGETF2(mat, n, k, k, blockSize)) {
            std::cerr << "Block LU decomposition failed." << std::endl;
            return false;
        }
        // DLASWP(...);
        // DTRSM(...);
        // DGEMM(...);
    }
    return true;
}

// correct
//bool parallelLU_Decomposition(std::vector<float>& mat, int n) {
//    const float smallVal = 1e-12;
//
//    for (int k = 0; k < n; ++k) {
//        if (std::fabs(mat[k*n + k]) < smallVal) {
//            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
//            return false;
//        }
//
//        std::vector<int> index(n - (k + 1));
//        std::iota(index.begin(), index.end(), k + 1);
//
//        std::for_each(std::execution::par, index.begin(), index.end(), [&](int i) {
//            mat[i*n + k] /= mat[k*n + k];
//            for (int j = k + 1; j < n; ++j) {
//                mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
//            }
//        });
//    }
//    return true;
//}

//correct method 1
//Doolittle algorithm
//DGETF2 by using partial pivoting
//bool DGETF2(std::vector<float>& mat, int n,int startRow, int startCol, int blockSize) {
//    const float smallVal = 1e-12; // define a small value to check if a number is close to zero
//
//    for (int k = 0; k < n; ++k) {
//        int pivot = IDAMAX(mat, n, k);
//        if (std::fabs(mat[pivot*n + k]) < smallVal) {
//            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
//            return false;
//        }
//        DSWAP(mat, n, k, pivot);
//
//        DSCAL(mat, n, k);
//        DGER(mat, n, k);
//    }
//    return true;
//}


int main() {
    int n = 4; // dimension of the matrix
    std::vector<float> A = {2, 1, 1,6,
                            4,-6, 0,8,
                            -2, 7, 2,9,
                            4, 8, 3, 1 }; // matrix to be decomposed

    // method 1
    std::vector<float> mat = A;
    if (!DGETF2(mat, n, 0, 0, n)) {
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

    return 0;
}



// receive a matrix and its dimension, and return the L and U matrices
//bool LU_Decomposition2(const std::vector<float>& A, std::vector<float>& L, std::vector<float>& U, int n);

//// correct method 2, no need. just return one matrix
//bool LU_Decomposition2(const std::vector<float>& A, std::vector<float>& L, std::vector<float>& U, int n) {
//    const float EPS = 1e-12;
//    L.resize(n * n, 0);
//    U.resize(n * n, 0);
//
//    for (int i = 0; i < n; ++i) {
//        L[i*n + i] = 1; // initialize the diagonal of L with 1
//    }
//
//    for (int k = 0; k < n; ++k) {
//        if (std::fabs(A[k*n + k]) < EPS) {
//            std::cerr << "Pivot element is close to zero. Cannot proceed." << std::endl;
//            return false;
//        }
//        for (int i = k; i < n; ++i) { // calculate U's row
//            float sum = 0;
//            for (int j = 0; j < k; ++j) {
//                sum += L[k*n + j] * U[j*n + i];
//            }
//            U[k*n + i] = A[k*n + i] - sum;
//        }
//        for (int i = k + 1; i < n; ++i) { // calculate L's column
//            float sum = 0;
//            for (int j = 0; j < k; ++j) {
//                sum += L[i*n + j] * U[j*n + k];
//            }
//            L[i*n + k] = (A[i*n + k] - sum) / U[k*n + k];
//        }
//    }
//    return true;
//}
