#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>

// DGETRF:
// for each A11 block{
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

// IDAMAX Correct!
// Find the row with the maximum value in the column, starting from the k downwards.

int IDAMAX(const std::vector<float>& mat, int blockLength, int BigDimension, int k, int blockStart) {
    int pivotRow = k;
    float maxVal = std::fabs(mat[k*BigDimension+k]);
    for (int i = k+1; i < blockStart+blockLength; ++i) {
        float val = std::fabs(mat[i*BigDimension + k]);
        if (val > maxVal) {
            maxVal = val;
            pivotRow = i;
        }
    }
    return pivotRow;
}

// DSWAP Correct!
// swap the data in the blockStartCol to blockStartCol+blockLength columns of row1 and row2
void DSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) {
    if (row1 == row2) return;
    for (int i = blockStartCol; i < endCol; ++i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    }
}

// Swap the elements in the P array
// Correct!
void SwapParray(std::vector<int>& pMat, int row1, int row2) {
    if (row1 == row2) return;
    std::swap(pMat[row1], pMat[row2]);
}

// DSCAL Correct!
// scale a column of a matrix
void DSCAL(std::vector<float>& mat, int n,int blockLength, int k, int blockStart) {
    // make k-th column but in a block all divided by the diagonal element
    for (int i = k+1; i < blockStart+blockLength; ++i) {
        mat[i*n + k] /= mat[k*n + k];
    }
}

// DGER Correct!
// perform a rank-k update on a matrix but only within a given range of rows and columns
void DGER(std::vector<float>& mat, int bigDimension, int k, int blockStart, int blockLength) {
    for (int i = k+1; i < blockStart+blockLength; ++i) {
        for (int j = k+1; j < blockStart+blockLength; ++j) {
            mat[i*bigDimension + j] -= mat[i*bigDimension + k] * mat[k*bigDimension + j];
        }
    }
}

// Apply row interchanges to the left and the right of the panel.
void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
    // create a vector to keep track of which rows have been swapped to avoid duplicate swaps

    // create a vector A, the values are from blockStart to blockStart+blockLength-1
    std::vector<int> A(blockLength);
    std::iota(A.begin(), A.end(), 0);

    for (int i = 0; i < A.size(); ++i) {
        // when A[i] is not in the correct position
        if (A[i]+blockStart != pMat[i+blockStart]) {
            // find the index of the element that should be swapped with A[i]
            int swapIndex = -1;
            for (int j = 0; j < A.size(); ++j) {
                if (A[j]+blockStart == pMat[i+blockStart]) {
                    swapIndex = j;
                    break;
                }
            }

            // actually no need to check swapIndex, because the swapIndex must be found
//            if (swapIndex != -1) {
            // swap the left part of the matrix
            DSWAP(mat, 0, blockStart, i+blockStart, swapIndex+blockStart, n);
            // swap the right part of the matrix
            DSWAP(mat, blockStart+blockLength, n, i+blockStart, swapIndex+blockStart, n);

            std::swap(A[i], A[swapIndex]);
//            }
        }
    }
}

// Apply row interchanges to the left and the right of the panel.
// this is totally correct, but the copy is a bad idea!
//void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
//    // create a size of blockLength*n matrix, then copy the values of mat to this new matrix,
//    // then according to the value of pMat, copy the values of the new matrix back to mat
//    std::vector<float> tempMat(blockLength*n, 0);
//    for (int i = 0; i < blockLength; ++i) {
//        for (int j = 0; j < n; ++j) {
//            tempMat[i*n + j] = mat[(blockStart+i)*n + j];
//        }
//    }
//    // copy the values of tempMat back to mat according to the value of pMat
//    for (int i = 0; i < blockLength; ++i) {
//        for (int j = 0; j < n; ++j) {
//            // if the value of pMat is the same as the row number, then no need to swap
//            // no need to swap the values in the block area
//            if (j < blockStart || j >= blockStart+blockLength && blockStart+i != pMat[blockStart+i]) {
//                mat[(blockStart+i)*n + j] = tempMat[(pMat[blockStart+i]-blockStart)*n + j];
//            }
//        }
//    }
//}

// compute U12(A12)
// A12 = L11^-1 * A12
void DTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    for (int i = blockStart; i < blockStart + blockLength; ++i) {
        for (int j = blockStart+blockLength; j < n; ++j) {
            float sum = 0.0;
            for (int k = blockStart; k < i; ++k) {
                sum += mat[i * n + k] * mat[k * n + j];
            }
            mat[i * n + j] -= sum;
        }
    }
}

// PA = LU
// Naive LU decomposition, calculate the L and U for a block in mat matrix
bool DGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) {
    for (int k = blockStartCol; k < blockStartCol+blockLength; ++k) {
        // find the pivot value in the k-th column and set that row index to pivotRow
        int pivotRow = IDAMAX(mat, blockLength, bigDimension,k, blockStartCol);
        // swap the pivot row with the kth row
        DSWAP(mat, blockStartCol, blockStartCol+blockLength, k, pivotRow, bigDimension);
        SwapParray(pMat, k, pivotRow);

        // calculate the elements of L and U
        DSCAL(mat, bigDimension, blockLength, k,blockStartCol);
        DGER(mat, bigDimension, k, blockStartCol, blockLength);
    }
    return true;
}

// L21 = A21 * U11^-1
void DTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    for (int i = blockStart + blockLength; i < n; ++i) {
        for (int j = blockStart; j < blockStart + blockLength; ++j) {
            float L21_value = mat[i * n + j];
            for (int k = blockStart; k < j; ++k) {
                L21_value -= mat[i * n + k] * mat[k * n + j];
            }
            if (mat[j * n + j] != 0) {
                mat[i * n + j] = L21_value / mat[j * n + j];
            } else {
                std::cerr << "Division by zero encountered in DTRSM_L21." << std::endl;
                return;
            }
        }
    }
}

// DGEMM to update A22
// A22 = A22 - L21 * U12
void DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    for (int i = blockStart + blockLength; i < n; ++i) {
        for (int j = blockStart + blockLength; j < n; ++j) {
            float sum = 0.0;
            for (int k = blockStart; k < blockStart + blockLength; ++k) {
                sum += mat[i * n + k] * mat[k * n + j];
            }
            mat[i * n + j] -= sum;
        }
    }
}


// BLOCK_DGEMM to update A22
// A22 = A22 - L21 * U12
//void BLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength) {
//    // can assume that smallBlockLength is equal to blockLength, which may be tested later
//    for (int i = blockStart + blockLength; i < n; ++i) {
//        for (int j = blockStart + blockLength; j < n; ++j) {
//            float sum = 0.0;
//            for (int k = blockStart; k < blockStart + blockLength; ++k) {
//                sum += mat[i * n + k] * mat[k * n + j];
//            }
//            mat[i * n + j] -= sum;
//        }
//    }
//}

void BLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength = 5) {
    // can assume that smallBlockLength is equal to blockLength, which may be tested later
    int A22start = blockStart + blockLength;
    for (int ii = A22start; ii < n; ii += smallBlockLength) {
        for (int jj = A22start; jj < n; jj += smallBlockLength) {
            for (int kk = blockStart; kk < A22start; kk += smallBlockLength) {
                for (int i = ii; i < std::min(ii + smallBlockLength, n); ++i) {
                    for (int j = jj; j < std::min(jj + smallBlockLength, n); ++j) {
                        float sum = 0.0;
                        for (int k = kk; k < std::min(kk + smallBlockLength, A22start); ++k) {
                            sum += mat[i * n + k] * mat[k * n + j];
                        }
                        mat[i * n + j] -= sum;
                    }
                }
            }
        }
    }
}

// Main function for LU decomposition!
bool DGETRF(std::vector<float>& mat, int n, int blockLength, std::vector<int>& pMat) {

    //A11 A12
    //A21 A22

    // when n=4, blockLength=2
    // whole matrix 4*4  block 2*2,
    // first loop A11 is 2*2 matrix
    //            A12 is 2*4 matrix
    //            A21 is 2*2 matrix
    //            A22 is 2*2 matrix
    // next loop applied to A22

    int blockStart = 0;
    for (; blockStart+blockLength <= n; blockStart+=blockLength) {

        // applied LU factorization to A11, output L11 and U11 and P11
        DGETF2(mat, pMat, blockLength, blockStart, n);

        // pMat example: 1 0 2 3  indicates swap row 0 and 1

        // DLASWP -  Apply row interchanges to the left and the right of this block.
        DLASWP(mat, pMat, n, blockStart, blockLength);

        // compute U12
        DTRSM(mat, n, blockStart, blockLength);

        // compute L21
        // L21 = A21 * U11^-1
        DTRSM_L21(mat, n, blockStart, blockLength);

        // update A22 here
        // A22 = A22 - L21 * U12
        BLOCK_DGEMM(mat, n, blockStart, blockLength);
    }
    // if there is a smaller block left
    if (blockStart < n) {
        DGETF2(mat, pMat, n-blockStart, blockStart, n);
        DLASWP(mat, pMat, n, blockStart, n-blockStart);
    }
    return true;
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

std::vector<float> matrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
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

int main() {

//    int n = 5; // dimension of the matrix
//
//    std::vector<float> A = {2, 1, 1, 0, 4,
//                            4, 3 ,3, 1, 5,
//                            8, 7, 9, 5, 6,
//                            6, 7, 9, 8, 8,
//                            9, 2, 3, 4, 5} ; // matrix to be decomposed

//    std::vector<float> A = {2, 1, 1, 0,
//                            4, 3 ,3, 1,
//                            8, 7, 9, 5,
//                            6, 7, 9, 8,
//                            } ; // matrix to be decomposed

    std::vector<int> sizes = {6,1,2,3,234,232,231,230,100,4,5,111,101};
    for (int n : sizes) {
        auto A = generateRandomMatrix(n, n);

        std::vector<float> mat = A;

        std::vector<int> pMat(n);
        std::iota(pMat.begin(), pMat.end(), 0);

        DGETRF(mat, n, 9, pMat);
//         print P
//        for (int i = 0; i < n; ++i) {
//            std::cout << pMat[i] << " ";
//        }

        std::vector<float> L(n * n, 0);
        std::vector<float> U(n * n, 0);
        for (int i = 0; i < n; ++i) {
            L[i * n + i] = 1;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i <= j) {
                    U[i * n + j] = mat[i * n + j];
                } else {
                    L[i * n + j] = mat[i * n + j];
                }
            }
        }
        std::vector<float> product = matrixMultiply(L, U, n, n, n, n);
        std::vector<float> result(n * n, 0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result[pMat[i] * n + j] = product[i * n + j];
            }
        }

        bool isEqual = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::fabs(result[i * n + j] - A[i * n + j]) > 0.5) {
                    std::cout << "result[" << i << "][" << j << "] = " << result[i * n + j] << ", A[" << i << "][" << j << "] = " << A[i * n + j] << std::endl;
                    isEqual = false;
                }
            }
        }
        if (isEqual) {
            std::cout << "Decomposition succeeded!" << std::endl;
        } else {
            std::cout << "Decomposition failed!" << std::endl;
        }
    }

    return 0;
}

//bool  Naive_LU_Decomposition(std::vector<float>& mat, int n) {
//    const float smallVal = 1e-12; // define a small value to check if a number is close to zero
//
//    for (int k = 0; k < n; ++k) {
//        for (int i = k + 1; i < n; ++i) {
//            mat[i*n + k] /= mat[k*n + k];
//            for (int j = k + 1; j < n; ++j) {
//                mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
//            }
//        }
//    }
//    return true;
//}
