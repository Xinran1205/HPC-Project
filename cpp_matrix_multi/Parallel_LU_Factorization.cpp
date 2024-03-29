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

// not easy to parallel,
// my idea is to divide this column into chunks and parallel find the max value in each chunk and then find the max value in the max values
int IDAMAX(const std::vector<float>& mat, int blockLength, int BigDimension, int k) {
    int pivotRow = k;
    float maxVal = std::fabs(mat[k*BigDimension+k]);
    for (int i = k+1; i < blockLength; ++i) {
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

// can be parallel!!!!
void DSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) {
    if (row1 == row2) return;
//    for (int i = blockStartCol; i < endCol; ++i) {
//        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
//    }

    std::vector<int> indices(endCol - blockStartCol);
    std::iota(indices.begin(), indices.end(), blockStartCol);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    });
}

// Swap the elements in the P array
// Correct!
void SwapParray(std::vector<int>& pMat, int row1, int row2) {
    if (row1 == row2) return;
    std::swap(pMat[row1], pMat[row2]);
}

// DSCAL Correct!
// scale a column of a matrix

// can be parallel!!!!
void DSCAL(std::vector<float>& mat, int n,int blockLength, int k, int blockStart) {
    // make k-th column but in a block all divided by the diagonal element
//    for (int i = k+1; i < blockStart+blockLength; ++i) {
//        mat[i*n + k] /= mat[k*n + k];
//    }

    int numElements = blockStart + blockLength - (k + 1);
    std::vector<int> indices(numElements);
    std::iota(indices.begin(), indices.end(), k + 1);
    float diagonalElement = mat[k * n + k];

    // every thread share the same diagonalElement
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        mat[i * n + k] /= diagonalElement;
    });
}

// DGER Correct!
// perform a rank-k update on a matrix but only within a given range of rows and columns

// can be parallel!!!!
void DGER(std::vector<float>& mat, int bigDimension, int k, int blockStart, int blockLength) {
//    for (int i = k+1; i < blockStart+blockLength; ++i) {
//        for (int j = k+1; j < blockStart+blockLength; ++j) {
//            mat[i*bigDimension + j] -= mat[i*bigDimension + k] * mat[k*bigDimension + j];
//        }
//    }
    std::vector<int> indices(blockStart+blockLength - (k + 1));
    std::iota(indices.begin(), indices.end(), k + 1);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        for (int j = k + 1; j < blockStart + blockLength; ++j) {
            mat[i * bigDimension + j] -= mat[i * bigDimension + k] * mat[k * bigDimension + j];
        }
    });
}

// cannot parallel!!!!!!
// Apply row interchanges to the left and the right of the panel.
void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
    // create a vector to keep track of which rows have been swapped to avoid duplicate swaps
    std::vector<int> isSwapped(pMat.size(), 0);
    // go through the P array, start index is blockStart
    for (int i = blockStart; i < blockStart+blockLength; ++i) {
        if (i != pMat[i] && isSwapped[i] != pMat[i]){
            // avoid swap the block part of the matrix
            // swap the left part of the matrix
            DSWAP(mat, 0, blockStart, i, pMat[i], n);
            // swap the right part of the matrix
            DSWAP(mat, blockStart+blockLength, n, i, pMat[i], n);

            // avoid swapping the same row twice
            isSwapped[i] = pMat[i];
            isSwapped[pMat[i]] = i;
        }
    }
}

// compute U12(A12)
// A12 = L11^-1 * A12

// might be parallel
void DTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
//    int offset = n - blockStart - blockLength;
//    for (int i = 0; i < blockLength; ++i) {
//        for (int j = 0; j < offset; ++j) {
//            float sum = 0.0;
//            for (int k = 0; k < i; ++k) {
//                sum += mat[(blockStart + i) * n + blockStart + k] *
//                       mat[(blockStart + k) * n + blockStart + blockLength + j];
//            }
//            mat[(blockStart + i) * n + blockStart + blockLength + j] -= sum;
//        }
//    }


    int offset = n - blockStart - blockLength;
    std::vector<int> indices(blockLength);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        // the sum variable is private to each thread
        for (int j = 0; j < offset; ++j) {
            float sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += mat[(blockStart + i) * n + blockStart + k] *
                       mat[(blockStart + k) * n + blockStart + blockLength + j];
            }
            mat[(blockStart + i) * n + blockStart + blockLength + j] -= sum;
        }
    });
}

// Computer L21(A21)
// L21 = A21 * U11^-1

// can be parallel, as same as above
void DTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) {
//    for (int i = blockStart + blockLength; i < n; ++i) {
//        for (int j = blockStart; j < blockStart + blockLength; ++j) {
//            float L21_value = mat[i * n + j];
//            for (int k = blockStart; k < j; ++k) {
//                L21_value -= mat[i * n + k] * mat[k * n + j];
//            }
//            if (mat[j * n + j] != 0) {
//                mat[i * n + j] = L21_value / mat[j * n + j];
//            }
//        }
//    }

        std::vector<int> indices(n - blockStart - blockLength);
        std::iota(indices.begin(), indices.end(), blockStart + blockLength);

        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
            for (int j = blockStart; j < blockStart + blockLength; ++j) {
                float L21_value = mat[i * n + j];
                for (int k = blockStart; k < j; ++k) {
                    L21_value -= mat[i * n + k] * mat[k * n + j];
                }
                if (mat[j * n + j] != 0) {
                    mat[i * n + j] = L21_value / mat[j * n + j];
                }
            }
        });
}

// DGEMM to update A22
// A22 = A22 - L21 * U12

// can be parallel, as same as above
void DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
//    for (int i = blockStart + blockLength; i < n; ++i) {
//        for (int j = blockStart + blockLength; j < n; ++j) {
//            float sum = 0.0;
//            for (int k = blockStart; k < blockStart + blockLength; ++k) {
//                sum += mat[i * n + k] * mat[k * n + j];
//            }
//            mat[i * n + j] -= sum;
//        }
//    }

    std::vector<int> indices(n - blockStart - blockLength);
    std::iota(indices.begin(), indices.end(), blockStart + blockLength);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        for (int j = blockStart + blockLength; j < n; ++j) {
            float sum = 0.0;
            for (int k = blockStart; k < blockStart + blockLength; ++k) {
                sum += mat[i * n + k] * mat[k * n + j];
            }
            mat[i * n + j] -= sum;
        }
    });
}

// Correct!! PA = LU
// Naive LU decomposition, calculate the L and U for a block in mat matrix
bool DGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) {
    for (int k = blockStartCol; k < blockStartCol+blockLength; ++k) {
        // find the pivot value in the k-th column and set that row index to pivotRow
        int pivotRow = IDAMAX(mat, blockLength, bigDimension,k);
        // swap the pivot row with the kth row
        DSWAP(mat, blockStartCol, blockStartCol+blockLength, k, pivotRow, bigDimension);
        SwapParray(pMat, k, pivotRow);

        // these two functions can be combined into one function! load once!
        DSCAL(mat, bigDimension, blockLength, k,blockStartCol);
        DGER(mat, bigDimension, k, blockStartCol, blockLength);
    }
    return true;
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
        DGEMM(mat, n, blockStart, blockLength);
    }
    // if there is a smaller block left
    if (blockStart < n) {
        DGETF2(mat, pMat, n-blockStart, blockStart, n);
        DLASWP(mat, pMat, n, blockStart, n-blockStart);
    }
    return true;
}

int main() {
    int n = 5; // dimension of the matrix

    std::vector<float> A = {2, 1, 1, 0, 4,
                            4, 3 ,3, 1, 5,
                            8, 7, 9, 5, 6,
                            6, 7, 9, 8, 8,
                            9, 2, 3, 4, 5} ; // matrix to be decomposed

//    std::vector<float> A = {2, 1, 1, 0,
//                            4, 3 ,3, 1,
//                            8, 7, 9, 5,
//                            6, 7, 9, 8,
//                            } ; // matrix to be decomposed

    std::vector<float> mat = A;

    std::vector<int> pMat(n);
    std::iota(pMat.begin(), pMat.end(), 0);

    DGETRF(mat, n, 3,pMat);

    std::cout << "Result of LU Decomposition (in-place):" << std::endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(8) << mat[i*n + j] << " ";
        }
        std::cout << std::endl;
    }

    std ::cout << "P array:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << pMat[i] << " ";
    }
    return 0;
}

//// correct
//bool parallelLU_Decomposition(std::vector<float>& mat, int n) {
//    for (int k = 0; k < n; ++k) {
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

//// DGEMM Correct!
//std::vector<float> parallelMatrixMultiply(const std::vector<float>& mat1, const std::vector<float>& mat2, int rows1, int cols1, int rows2, int cols2) {
//    if (rows1 == 0 || cols1 == 0 || rows2 == 0 || cols2 == 0 || cols1 != rows2) {
//        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
//    }
//
//    std::vector<float> product(rows1 * cols2, 0.0f);
//
//    std::vector<int> index(rows1);
//    std::iota(index.begin(), index.end(), 0);
//
//    std::for_each(std::execution::par, index.begin(), index.end(), [cols1, cols2, &mat1, &mat2, &product](int i) {
//        for (int j = 0; j < cols2; ++j) {
//            for (int k = 0; k < cols1; ++k) {
//                product[i * cols2 + j] += mat1[i * cols1 + k] * mat2[k * cols2 + j];
//            }
//        }
//    });
//
//    return product;
//}