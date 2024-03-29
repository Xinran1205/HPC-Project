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
//void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
//    // create a vector to keep track of which rows have been swapped to avoid duplicate swaps
//    std::vector<int> isSwapped(pMat.size(), 0);
//    // go through the P value related to this block
//    for (int i = blockStart; i < blockStart+blockLength; ++i) {
//        if (i != pMat[i] && isSwapped[i] != pMat[i]){
//            //avoid swap the block part of the matrix
//            // swap the left part of the matrix
//            DSWAP(mat, 0, blockStart, i, pMat[i], n);
//            // swap the right part of the matrix
//            DSWAP(mat, blockStart+blockLength, n, i, pMat[i], n);
//
//            // avoid swapping the same row twice
//            isSwapped[i] = pMat[i];
//            isSwapped[pMat[i]] = i;
//        }
//    }
//}

// Apply row interchanges to the left and the right of the panel.
// 这个应该是完全正确的，但是这个拷贝非常恶心人！
void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
    // 创建一个大小是blocklength*n的新矩阵，然后把mat的值拷贝到这个新矩阵中，然后再根据pMat的值，把新矩阵的值拷贝回mat中
    std::vector<float> tempMat(blockLength*n, 0);
    for (int i = 0; i < blockLength; ++i) {
        for (int j = 0; j < n; ++j) {
            tempMat[i*n + j] = mat[(blockStart+i)*n + j];
        }
    }
    // 根据pMat的值，把tempMat中的值拷贝回mat中
    for (int i = 0; i < blockLength; ++i) {
        for (int j = 0; j < n; ++j) {
            // 需要加入判断，不要动block区域的值，并且如果pMat中的值和现在的行号相同也不用动
            if (j < blockStart || j >= blockStart+blockLength && blockStart+i != pMat[blockStart+i]) {
                mat[(blockStart+i)*n + j] = tempMat[(pMat[blockStart+i]-blockStart)*n + j];
            }
        }
    }
}

// compute U12(A12)
// A12 = L11^-1 * A12
void DTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    int offset = n - blockStart - blockLength;
    for (int i = 0; i < blockLength; ++i) {
        for (int j = 0; j < offset; ++j) {
            float sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += mat[(blockStart + i) * n + blockStart + k] *
                       mat[(blockStart + k) * n + blockStart + blockLength + j];
            }
            mat[(blockStart + i) * n + blockStart + blockLength + j] -= sum;
        }
    }
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

        // calculate the elements of L and U
        DSCAL(mat, bigDimension, blockLength, k,blockStartCol);
        DGER(mat, bigDimension, k, blockStartCol, blockLength);
    }
    return true;
}

// Computer L21(A21)
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
//        // 当blockStart=0时，打印P的前3个
//        if (blockStart == 0) {
//            for (int i = 0; i < 3; ++i) {
//                std::cout << pMat[i] << " ";
//            }
//            std::cout << std::endl;
//        }

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

    std::vector<int> sizes = {7,2,3,1,9,10,24,52,70,100};
    for (int n : sizes) {
        auto A = generateRandomMatrix(n, n);
//        // 打印A
//        for (int i = 0; i < n; ++i) {
//            for (int j = 0; j < n; ++j) {
//                std::cout << std::setw(10) << A[i * n + j] << " ";
//            }
//            std::cout << std::endl;
//        }

        std::vector<float> mat = A;

        std::vector<int> pMat(n);
        std::iota(pMat.begin(), pMat.end(), 0);

        DGETRF(mat, n, 3, pMat);
        // 我需要在这里写一个自动化校验，新创建两个矩阵，一个是L，一个是U，大小都和A一样，然后把mat中的值分别填入L和U中，然后再把L和U相乘，
        // 然后把相乘的结果根据P矩阵还原成原来的矩阵，然后和A比较，如果相等，说明分解正确

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
        // 根据p的这一行的值，把product中的值还原到result中
        // pMat example: 1 0 2 3  indicates swap row 0 and 1
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result[pMat[i] * n + j] = product[i * n + j];
            }
        }

        // 比较result和A是否相等，for循环比较
        bool isEqual = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // 这里可能会存在一些误差！
                if (std::fabs(result[i * n + j] - A[i * n + j]) > 1) {
                    // 打印不相等的值
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
