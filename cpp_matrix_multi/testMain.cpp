#include "Matrix.hpp"
#include "SparseMatrixCOO.hpp"
#include "SparseMatrixCSR.hpp"

std::vector<std::vector<float>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distrib(0.0f, 100.0f);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = distrib(gen);
        }
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
//    test transpose
//    std::vector<std::vector<int>> vec{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    Matrix<int> m4(vec);
//    Matrix<int> m5;
//    m5 = m4.transpose();
//    std::cout << m5 << std::endl;
//
//    std ::cout << m4 << std::endl;

//    test block multiplication
//    std::vector<int> sizes = {900};
//    for (int n : sizes) {
//        auto mat1 = generateRandomMatrix(n, n);
//        auto mat2 = generateRandomMatrix(n, n);
//
//        // check if the block matrix multiplication is correct
//        Matrix<float> A1(mat1);
//        Matrix<float> A2(mat2);
//        Matrix<float> A3;
//        Matrix<float> A4;
//        Matrix<float> A5;
//        A4 = A1 * A2;
//        A3 = A1.blockMultiplication(A2, 3);
//        A5 = A1.parallelBlockMultiply(A2, 3);
//
//        if (A3 == A4 && A3 == A5) {
//            std::cout << "Block matrix multiplication is correct!" << std::endl;
//        } else {
//            std::cerr << "Block matrix multiplication is incorrect!" << std::endl;
//        }
//        std::cout << std::endl;
//    }

    // test LU decomposition
    std::vector<int> sizes = {6,1,2,5,23,45,67,98,239,850,112,523};
    for (int n : sizes) {
        auto A = generateRandomMatrix(n, n);

        Matrix<float> mat(A);

//        Matrix<float> result = mat.LU_Factorization(5);
        Matrix<float> result = mat.Parallel_LU_Factorization(5);
        std::vector<float> L(n * n, 0);
        std::vector<float> U(n * n, 0);
        for (int i = 0; i < n; ++i) {
            L[i * n + i] = 1;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i <= j) {
                    U[i * n + j] = result(i, j);
                } else {
                    L[i * n + j] = result(i, j);
                }
            }
        }
        std::vector<float> product = matrixMultiply(L, U, n, n, n, n);
        std::vector<float> result2(n * n, 0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result2[result.getP()[i] * n + j] = product[i * n + j];
//                result[pMat[i] * n + j] = product[i * n + j];
            }
        }

        bool isEqual = true;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if ( std::fabs(result2[i * n + j] - A[i][j]) > 0.5) {
                    std::cout << "result[" << i << "][" << j << "] = " << result2[i * n + j] << ", A[" << i << "][" << j
                              << "] = " << A[i][j] << std::endl;
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

// test sparse matrix
    SparseMatrixCOO<int> matrix1 = {
            {0, 0, 3, 0, 0},
            {4, 0, 0, 0, 0},
            {0, 5, 6, 0, 0},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 7, 0}
    };
    SparseMatrixCOO<int> matrix2 = {
            {1, 2, 3, 0, 0},
            {4, 0, 0, 0, 0},
            {0, 5, 6, 8, 0},
            {0, 0, 0, 0, 0},
            {0, 0, 0, 7, 0}
    };

    SparseMatrixCOO<int> matrix2_transpose = matrix2.transpose();
    matrix2_transpose.printNonZeroElements();

    SparseMatrixCOO<int> result1 = matrix1 + matrix2;
    // print the non-zero elements of the result matrix
    result1.printNonZeroElements();

// test sparse matrix CSR
    SparseMatrixCSR<int> matrix3({
                                        {1, 0, 0, 0},
                                        {2, 0, 0, 0},
                                        {0, 0, 3, 0},
                                        {0, 0, 0, 4}
                                });

    SparseMatrixCSR<int> matrix4({
                                         {1, 0, 0, 0},
                                         {0, 2, 0, 0},
                                         {9, 0, 3, 0},
                                         {0, 0, 5, 4}
                                 });

//    matrix.printNonZeroElements();
//    matrix2.printNonZeroElements();

    SparseMatrixCSR<int> result2 = matrix3 + matrix4;
    result2.printNonZeroElements();

    return 0;
}
