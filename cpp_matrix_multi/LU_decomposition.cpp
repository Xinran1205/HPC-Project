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
// swap two rows of a matrix but only within a given range of columns
// 交换row1和row2两行的从blockStartCol开始到blockStartCol+blockLength列的数据
void DSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) {
    if (row1 == row2) return;
    for (int i = blockStartCol; i < endCol; ++i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    }
}

// 把pMat中的row1和row2两行的数据交换
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


void DLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) {
    std::vector<int> isSwapped(pMat.size(), 0);
    // go through the P value related to this block
    for (int i = blockStart; i < blockStart+blockLength; ++i) {
        if (i != pMat[i] && isSwapped[i] != pMat[i]){
            //avoid swap the block part of the matrix
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

// compute U12  这个函数在3*3时有bug
//  A12 = L11^-1 * A12
void DTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    int offset = n - blockStart - blockLength; // 减少重复计算

    // 直接在原始矩阵上操作以计算L11^-1 * A12，省略了临时L11和A12的构建
    for (int i = 0; i < blockLength; ++i) {
        for (int j = 0; j < offset; ++j) {
            float sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += mat[(blockStart + i) * n + blockStart + k] *
                       mat[(blockStart + k) * n + blockStart + blockLength + j];
            }
            mat[(blockStart + i) * n + blockStart + blockLength + j] -= sum;
            // 假定对角线元素不为0（通常在LU分解中确保），无需每次检查
        }
    }

    // 不需要单独的步骤来复制A12回mat，因为已经直接在mat上操作
}

bool LU_Decomposition(std::vector<float>& mat, int n) {
    const float smallVal = 1e-12; // define a small value to check if a number is close to zero

    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            mat[i*n + k] /= mat[k*n + k];
            for (int j = k + 1; j < n; ++j) {
                mat[i*n + j] -= mat[i*n + k] * mat[k*n + j];
            }
        }
    }
    return true;
}

// Correct!! PA = LU
bool DGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) {
    // k是列号
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

void DTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    // 对于A21中的每一行
    for (int i = blockStart + blockLength; i < n; ++i) {
        // 对于U11中的每一列
        for (int j = blockStart; j < blockStart + blockLength; ++j) {
            // 初始时假设L21的值就是A21的原始值
            float L21_value = mat[i * n + j];

            // 通过U11的当前列之前的所有列进行迭代来更新L21的值
            // 注意这里我们假设U11为上三角矩阵
            for (int k = blockStart; k < j; ++k) {
                L21_value -= mat[i * n + k] * mat[k * n + j];
            }

            // 分母为U11当前列的对角线元素，用于完成U11^-1的计算
            if (mat[j * n + j] != 0) { // 确保不会除以0
                mat[i * n + j] = L21_value / mat[j * n + j];
            } else {
                std::cerr << "Division by zero encountered in DTRSM_L21." << std::endl;
                return;
            }
        }
    }
}

void UpdateA22(std::vector<float>& mat, int n, int blockStart, int blockLength) {
    // 对于A22的每一行
    for (int i = blockStart + blockLength; i < n; ++i) {
        // 对于A22的每一列
        for (int j = blockStart + blockLength; j < n; ++j) {
            // 计算L21 * U12的结果并从A22中减去
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

        // pMat example 1 0 2 3  indicates swap row 0 and 1

        // DLASWP -  Apply row interchanges to the left and the right of this block.
        DLASWP(mat, pMat, n, blockStart, blockLength);

//         Compute U12
        DTRSM(mat, n, blockStart, blockLength);

        // compute L21
        // L21 = A21 * U11^-1
        DTRSM_L21(mat, n, blockStart, blockLength);

        // update A22 here
        // A22 = A22 - L21 * U12
        UpdateA22(mat, n, blockStart, blockLength);
//            DGEMM(...);
    }
    if (blockStart == n) {
        return true;
    }
    DGETF2(mat, pMat, n-blockStart, blockStart, n);
    DLASWP(mat, pMat, n, blockStart, n-blockStart);
    return true;
}

// correct
bool parallelLU_Decomposition(std::vector<float>& mat, int n) {
    for (int k = 0; k < n; ++k) {
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

int main() {
    int n = 4; // dimension of the matrix
//    std::vector<float> A = {2, 1,
//                            4, 3 }; // matrix to be decomposed

    std::vector<float> A = {2, 1, 1, 0,
                            4, 3 ,3, 1,
                            8, 7, 9, 5,
                            6, 7, 9, 8} ; // matrix to be decomposed

    // method 1
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

//    std::vector<int> pMat(n);
//    std::iota(pMat.begin(), pMat.end(), 0);

//    DGETF2(mat, pMat, 4, 0, n);
//    std::cout << "Result of LU Decomposition (in-place):" << std::endl;
//    for (int i = 0; i < n; ++i) {
//        for (int j = 0; j < n; ++j) {
//            std::cout << std::setw(8) << mat[i*n + j] << " ";
//        }
//        std::cout << std::endl;
//    }
//    std ::cout << "P array:" << std::endl;
//    for (int i = 0; i < n; ++i) {
//        std::cout << pMat[i] << " ";
//    }


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
