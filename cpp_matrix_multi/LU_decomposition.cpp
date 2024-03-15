#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>

// DGETRF:
// for each A11 like block{
//      DGETF2:{
//          IDAMAX:
//          DSWAP:
//          DSCAL:
//          DGER:
//          }
//  DLASWP:
//  DTRSM:
//  DGEMM:
// }


// DGEMM Correct!
std::vector<float> parallelMatrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    std::vector<float> product(rows1 * cols2, 0.0f);

    std::vector<int> index(rows1);
    std::iota(index.begin(), index.end(), 0);

    std::for_each(std::execution::par, index.begin(), index.end(), [cols1, cols2, &mat1, &mat2, &product](int i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
            }
        }
    });

    return product;
}

// IDAMAX Correct!
// Find the row with the maximum value in the k-th column, starting from the k-th row downwards.
int IDAMAX(const std::vector<float>& mat, int n, int k) {
    int pivotRow = k;
    float maxVal = std::fabs(mat[k*n+k]);
    for (int i = k+1; i < n; ++i) {
        float val = std::fabs(mat[i*n + k]);
        if (val > maxVal) {
            maxVal = val;
            pivotRow = i;
        }
    }
    return pivotRow;
}


// DSWAP Correct!
// swap two rows of a matrix but only within a given range of columns
void DSWAP(std::vector<float>& mat, int n, int row1, int row2) {
    if (row1 == row2) return;
    for (int i = 0; i < n; ++i) {
        std::swap(mat[row1*n + i], mat[row2*n + i]);
    }
}

// Swap the rows of a permutation matrix
// the size of P matrix is blockLength*blockLength
// Correct!
void SwapPmatrix(std::vector<int>& pMat, int blockLength, int row1, int row2) {
    if (row1 == row2) return;
    for (int i = 0; i < blockLength; ++i) {
        std::swap(pMat[row1 * blockLength + i], pMat[row2 * blockLength + i]);
    }
}

// DSCAL Correct!
// scale a column of a matrix
void DSCAL(std::vector<float>& mat, int n, int k) {
    // make k-th column all divided by the
    for (int i = k+1; i < n; ++i) {
        mat[i*n + k] /= mat[k*n + k];
    }
}

// DGER Correct!
// perform a rank-k update on a matrix but only within a given range of rows and columns
void DGER(std::vector<float>& mat, int n, int k) {
    for (int i = k+1; i < n; ++i) {
        for (int j = k+1; j < n; ++j) {
            mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
        }
    }
}

// Initialize P matrix as an identity matrix
void InitP(std::vector<int>& pMat, int n) {
    pMat.resize(n * n, 0);
    for (int i = 0; i < n; ++i) {
        pMat[i * n + i] = 1;
    }
}

// correct!
void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockIndex, int blockLength) {
    for (int i = 0; i < blockLength; ++i) {
        for (int j = 0; j < blockLength; ++j) {
            if (pMat[i * blockLength + j] == 1 && i != j) {
                DSWAP(mat, n, i+blockIndex*blockLength, j+blockIndex*blockLength);
                // this step is to avoid swap back
                SwapPmatrix(pMat, blockLength, i, j);
            }
        }
    }
}

void DTRSM(std::vector<float>& mat, const std::vector<float>& blockMat, int n, int blockIndex, int blockLength) {
    for (int i = 0; i < blockLength; ++i) {
        for (int j = blockIndex + blockLength; j < n; ++j) {
            // U12 = (L11)^-1 * A12
            float U12_value = mat[(blockIndex + i) * n + j];
            for (int k = 0; k < i; ++k) {
                U12_value -= blockMat[i * blockLength + k] * mat[(blockIndex + k) * n + j];
            }
            mat[(blockIndex + i) * n + j] = U12_value;
        }
    }
}

// Correct!! PAx=b
bool DGETF2(std::vector<float>& mat,std::vector<int>& pMat, int n) {
    // Initialize P matrix as an identity matrix
    InitP(pMat, n);

    for (int k = 0; k < n; ++k) {
        // find the pivot row
        int pivotRow = IDAMAX(mat, n, k);
        // swap the pivot row with the current row
        DSWAP(mat, n, k, pivotRow);
        SwapPmatrix(pMat, n, k, pivotRow);

        // calculate the elements of L and U
        DSCAL(mat, n, k);
        DGER(mat, n, k);
    }
    return true;
}

// Main function for LU decomposition!
bool DGETRF(std::vector<float>& mat, int n, int blockLength) {

    //A11 A12
    //A21 A22

    // when n=6, blockLength=2
    // whole matrix 6*6  block 2*2,
    // first loop A11 is 2*2 matrix
    //            A12 is 2*4 matrix
    //            A21 is 4*2 matrix
    //            A22 is 4*4 matrix

    // next loop applied to A22

    // assume it does not have remainder
    for (int blockIndex = 0; blockIndex < n; blockIndex+=blockLength) {
        // each time initialize a new P matrix for each block
        std::vector<int> pMat;
        // create a block matrix and copy the elements of the original matrix into it
        std::vector<float> blockMat(blockLength * blockLength);
        //initialize the block matrix
        for (int i = 0; i < blockLength; ++i) {
            for (int j = 0; j < blockLength; ++j) {
                blockMat[i * blockLength + j] = mat[(blockIndex + i) * n + blockIndex + j];
            }
        }

        // applied LU factorization to A11, output L11 and U11 and P11
        DGETF2(blockMat, pMat, blockLength);
        // blockMat is composed of L11 and U11

        // pMat example  0 1 0
        //               1 0 0
        //               0 0 1

        // DLASWP - apply the permutation matrix to the original matrix
        DLASWP(mat, pMat, n, blockIndex, blockLength);

        // update the original matrix with the block matrix
        for (int i = 0; i < blockLength; ++i) {
            for (int j = 0; j < blockLength; ++j) {
                mat[(blockIndex + i) * n + blockIndex + j] = blockMat[i * blockLength + j];
            }
        }

        // Compute U12
        DTRSM(mat, blockMat, n, blockIndex, blockLength);
        // how to compute L21


        // update A22 here
        // A22 = A22 - L21 * U12
//            DGEMM(...);
    }
    return true;
}

int main() {
    int n = 6; // dimension of the matrix
    std::vector<float> A = {2, 1, 1, 0, 4, 6,
                            4, 3 ,3, 1, 5, 2,
                            8, 7, 9, 5, 7, 1,
                            6, 7, 9, 8, 9, 2,
                            2, 4, 2, 12, 14, 5,
                            2, 3, 7, 8, 23, 12}; // matrix to be decomposed

    // method 1
    std::vector<float> mat = A;
//    std ::vector<int> pMat = {0, 1,
//                              1, 0};
//
//    DLASWP(mat, pMat, n, 0, 1);



//    std::cout << "Result of LU Decomposition (in-place):" << std::endl;
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << std::setw(8) << mat[i*n + j] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "P matrix:" << std::endl;
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << std::setw(8) << pMat[i*n + j] << " ";
//        }
//        std::cout << std::endl;
//    }
//
//    // block method
//    std::vector<float> mat2 = A;
//    DGETRF(mat2, n, 2);
//
//    std::cout << "Result of LU Decomposition (block method):" << std::endl;
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << std::setw(8) << mat2[i*n + j] << " ";
//        }
//        std::cout << std::endl;
//    }

    return 0;
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
