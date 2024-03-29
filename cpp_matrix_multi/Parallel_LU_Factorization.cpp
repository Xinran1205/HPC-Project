#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>
#include <thread>

// Find the row with the maximum value in the column, starting from the k downwards.
// not easy to parallel,
// my idea is to divide this column into chunks and parallel find the max value in each chunk and then find the max value in the max values

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

// swap the data in the blockStartCol to blockStartCol+blockLength columns of row1 and row2
void PDSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) {
    if (row1 == row2) return;
    std::vector<int> indices(endCol - blockStartCol);
    std::iota(indices.begin(), indices.end(), blockStartCol);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    });
}

// Swap the elements in the P array
void SwapParray(std::vector<int>& pMat, int row1, int row2) {
    if (row1 == row2) return;
    std::swap(pMat[row1], pMat[row2]);
}

// scale a column of a matrix
void PDSCAL(std::vector<float>& mat, int n,int blockLength, int k, int blockStart) {
    int numElements = blockStart + blockLength - (k + 1);
    std::vector<int> indices(numElements);
    std::iota(indices.begin(), indices.end(), k + 1);
    float diagonalElement = mat[k * n + k];

    // every thread share the same diagonalElement
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        mat[i * n + k] /= diagonalElement;
    });
}

// perform a rank-k update on a matrix but only within a given range of rows and columns
void PDGER(std::vector<float>& mat, int bigDimension, int k, int blockStart, int blockLength) {
    std::vector<int> indices(blockStart+blockLength - (k + 1));
    std::iota(indices.begin(), indices.end(), k + 1);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
        for (int j = k + 1; j < blockStart + blockLength; ++j) {
            mat[i * bigDimension + j] -= mat[i * bigDimension + k] * mat[k * bigDimension + j];
        }
    });
}

// This is not a good algorithm, because it copies the whole block (A12,left and right of the A11 block) to a temporary
// matrix and then copies it back.
// Should use line swap, but not easy to implement
void PDLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
    std::vector<float> tempMat(blockLength*n, 0);
    std::vector<int> indices(blockLength);
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i){
        for (int j = 0; j < n; ++j) {
            tempMat[i*n + j] = mat[(blockStart+i)*n + j];
        }
    });

    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i){
        for (int j = 0; j < n; ++j) {
            if (j < blockStart || j >= blockStart+blockLength && blockStart+i != pMat[blockStart+i]) {
                mat[(blockStart+i)*n + j] = tempMat[(pMat[blockStart+i]-blockStart)*n + j];
            }
        }
    });
}

// compute U12(A12)
// A12 = L11^-1 * A12

// cannot parallelize i, because the calculation of U12, for example, the second row or the third row,
// they all depend on the values of the previous rows
// but can parallelize j
void PDTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    std::vector<int> indicesJ(n - blockStart - blockLength);
    std::iota(indicesJ.begin(), indicesJ.end(), blockStart + blockLength);
    for (int i = blockStart; i < blockStart + blockLength; ++i) {
        std::for_each(std::execution::par, indicesJ.begin(), indicesJ.end(), [&](int j) {
            float sum = 0.0;
            for (int k = blockStart; k < i; ++k) {
                sum += mat[i * n + k] * mat[k * n + j];
            }
            mat[i * n + j] -= sum;
        });
    }
}

// Computer L21(A21)
// L21 = A21 * U11^-1

// this is the opposite of U12, i can be parallelized, but j cannot be parallelized
void PDTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) {
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

void PDGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    std::vector<int> indices(n - blockStart - blockLength);
    std::iota(indices.begin(), indices.end(), blockStart + blockLength);

    // this parallel will not have any problem, because mat[i * n + j] -= sum; will not affect the value of mat in the loop above
    // because the corresponding area is A22.
    // sum += mat[i * n + k] * mat[k * n + j]; the value taken here is from L21 and U12
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

// PA = LU
// Naive LU decomposition, calculate the L and U for a block in mat matrix
bool DGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) {
    for (int k = blockStartCol; k < blockStartCol+blockLength; ++k) {
        // find the pivot value in the k-th column and set that row index to pivotRow
        int pivotRow = IDAMAX(mat, blockLength, bigDimension,k, blockStartCol);
        // swap the pivot row with the kth row
//        std :: cout << "k: " << k << ", pivotRow: " << pivotRow << std::endl;
        PDSWAP(mat, blockStartCol, blockStartCol+blockLength, k, pivotRow, bigDimension);
        SwapParray(pMat, k, pivotRow);

        // these two functions can be combined into one function! load once!
        PDSCAL(mat, bigDimension, blockLength, k,blockStartCol);
        PDGER(mat, bigDimension, k, blockStartCol, blockLength);
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
        PDLASWP(mat, pMat, n, blockStart, blockLength);

        // compute U12
        PDTRSM(mat, n, blockStart, blockLength);

        // compute L21
        // L21 = A21 * U11^-1
        PDTRSM_L21(mat, n, blockStart, blockLength);

        // update A22 here
        // A22 = A22 - L21 * U12
        PDGEMM(mat, n, blockStart, blockLength);
    }
    // if there is a smaller block left
    if (blockStart < n) {
        DGETF2(mat, pMat, n-blockStart, blockStart, n);
        PDLASWP(mat, pMat, n, blockStart, n-blockStart);
    }
    return true;
}

// Function for checking the correctness of the LU decomposition
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

// Function for checking the correctness of the LU decomposition
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
//    int n = 4; // dimension of the matrix

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

    std::vector<int> sizes = {231,1,2,3,102,101,103,200};
    for (int n : sizes) {
        auto A = generateRandomMatrix(n, n);
//        // print A
//        for (int i = 0; i < n; ++i) {
//            for (int j = 0; j < n; ++j) {
//                std::cout << std::setw(10) << A[i * n + j] << " ";
//            }
//            std::cout << std::endl;
//        }
        std::cout << std::endl;

        std::vector<float> mat = A;

        std::vector<int> pMat(n);
        std::iota(pMat.begin(), pMat.end(), 0);

        DGETRF(mat, n, 3, pMat);

        // create two new matrices, one is L and the other is U, both the same size as A,
        // then fill the values in mat into L and U respectively,
        // then multiply L and U, then restore the result according to the P matrix to the original matrix,

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

        // pMat example: 1 0 2 3  indicates swap row 0 and 1
        // according to the value of this row in p, restore the value in product to result
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result[pMat[i] * n + j] = product[i * n + j];
            }
        }
//        // print result
//        for (int i = 0; i < n; ++i) {
//            for (int j = 0; j < n; ++j) {
//                std::cout << std::setw(10) << result[i * n + j] << " ";
//            }
//            std::cout << std::endl;
//        }

        // check the correctness of the decomposition
        bool isEqual = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // this might have some precision problem
                if (std::fabs(result[i * n + j] - A[i * n + j]) > 0.5) {
                    // print the unequal values
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