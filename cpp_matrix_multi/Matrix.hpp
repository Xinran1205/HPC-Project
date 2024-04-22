#ifndef CPP_MATRIX_MULTI_GENERALMATRIXCLASS_H
#define CPP_MATRIX_MULTI_GENERALMATRIXCLASS_H

#include "IMatrix.h"

//is_arithmetic ensure that the type is a numeric type
template<typename T>
class Matrix : public IMatrix<T> {
public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, T fillValue);
    Matrix(size_t rows, size_t cols, const std::vector<T>& data);
    Matrix(const std::vector<std::vector<T>>& data);
    Matrix(std::initializer_list<std::initializer_list<T>> list);

    bool operator==(const Matrix<T>& other) const;
    bool operator!=(const Matrix<T>& other) const;

    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;
    void printNonZeroElements() const override;
    size_t getRows() const override;
    size_t getCols() const override;
    const std::vector<T>& getData() const;
    const std::vector<int>& getP() const;
    void setP(std::vector<int> p);

    Matrix<T> transpose() const;
    Matrix<T> multiplyParallel(const Matrix<T>& other) const;
    Matrix<T> blockMultiplication(const Matrix<T>& other, int blockSize = 2) const;
    Matrix<T> parallelBlockMultiply(const Matrix<T>& other, int blockSize = 3) const;

    // Test the size of the block later, and the performance may be best when the two kinds of blocks are the same size!
    Matrix<T> LU_Factorization(int blockLength = 3,int smallBlockSize = 2) const;
    Matrix<T> Parallel_LU_Factorization(int blockLength = 3,int smallBlockSize = 2) const;

private:
    size_t m_rows, m_cols;
    std::vector<T> m_data;
    std::vector<int> pMat;

    int IDAMAX(const std::vector<T> &mat, int blockLength, int BigDimension, int k, int blockStart) const;
    void SwapParray(std::vector<int>& pMat, int row1, int row2) const;
    void DGETRF_intern(std::vector<T>& mat, int n, int blockLength, std::vector<int>& pMat, int smallBlockSize) const;
//    void DGEMM(std::vector<T> &mat, int n, int blockStart, int blockLength) const;
    void  BLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength) const;
    void DTRSM_L21(std::vector<T> &mat, int n, int blockStart, int blockLength) const;
    bool DGETF2(std::vector<T> &mat, std::vector<int> &pMat, int blockLength, int blockStartCol, int bigDimension) const;
    void DTRSM(std::vector<T> &mat, int n, int blockStart, int blockLength) const;
    void DLASWP(std::vector<T> &mat, std::vector<int> &pMat, int n, int blockStart, int blockLength) const;
    void DGER(std::vector<T> &mat, int bigDimension, int k, int blockStart, int blockLength) const;
    void DSCAL(std::vector<T> &mat, int n, int blockLength, int k, int blockStart) const;
    void DSWAP(std::vector<T> &mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) const;

    void PDGETRF_intern(std::vector<float>& mat, int n, int blockLength, std::vector<int>& pMat, int smallBlockSize) const;
    bool PDGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) const;
    void PBLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength) const;
//    void PDGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength) const;
    void PDTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) const;
    void PDTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) const;
    void PDLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) const;
    void PDGER(std::vector<float>& mat, int bigDimension, int k, int blockStart, int blockLength) const;
    void PDSCAL(std::vector<float>& mat, int n,int blockLength, int k, int blockStart) const;
    void PDSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) const;

    // here are the friend functions
friend Matrix<T> operator+(const Matrix<T>& mat1, const Matrix<T>& mat2) {
    if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(mat1.getRows(), mat1.getCols());
    for (size_t i = 0; i < mat1.getRows(); ++i) {
        for (size_t j = 0; j < mat1.getCols(); ++j) {
            result(i, j) = mat1(i, j) + mat2(i, j);
        }
    }

    return result;
}

friend Matrix<T> operator-(const Matrix<T>& mat1, const Matrix<T>& mat2) {
    if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(mat1.getRows(), mat1.getCols());
    for (size_t i = 0; i < mat1.getRows(); ++i) {
        for (size_t j = 0; j < mat1.getCols(); ++j) {
            result(i, j) = mat1(i, j) - mat2(i, j);
        }
    }

    return result;
}


friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.getRows(); ++i) {
        for (size_t j = 0; j < mat.getCols(); ++j) {
            os << std::setw(3) << mat(i, j) << " ";
        }
        os << std::endl;
    }

    os << "Matrix size: " << mat.getRows() << "x" << mat.getCols() << std::endl;
    return os;
}

// overload the multiplication operator
// naive matrix multiplication
friend Matrix<T> operator*(const Matrix<T>& mat1, const Matrix<T>& mat2) {
    if (mat1.getCols() != mat2.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(mat1.getRows(), mat2.getCols());
    for (size_t i = 0; i < mat1.getRows(); ++i) {
        for (size_t j = 0; j < mat2.getCols(); ++j) {
            for (size_t k = 0; k < mat1.getCols(); ++k) {
                result(i, j) += mat1(i, k) * mat2(k, j);
            }
        }
    }

    return result;
}
};

// Implementation of the public functions

template<typename T>
Matrix<T>::Matrix() : m_rows(0), m_cols(0) {}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, T()) {}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, T fillValue) : m_rows(rows), m_cols(cols), m_data(rows * cols, fillValue) {}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const std::vector<T>& data) : m_rows(rows), m_cols(cols), m_data(data) {
    if (data.size() != rows * cols) throw std::invalid_argument("Data size does not match matrix dimensions");
}

template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& data) : m_rows(data.size()), m_cols(data.empty() ? 0 : data[0].size()) {
    for (const auto& row : data) {
        if (row.size() != m_cols) throw std::invalid_argument("Irregular matrix shape");
        m_data.insert(m_data.end(), row.begin(), row.end());
    }
}

template<typename T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> list) : m_rows(list.size()), m_cols(list.begin()->size()) {
    for (const auto& row : list) {
        if (row.size() != m_cols) throw std::invalid_argument("Irregular matrix shape");
        m_data.insert(m_data.end(), row.begin(), row.end());
    }
}

template<typename T>
bool Matrix<T>:: operator==(const Matrix<T>& other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }
    for (size_t i = 0; i < m_rows * m_cols; ++i) {
        if (m_data[i] != other.m_data[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>:: operator!=(const Matrix<T>& other) const {
    return !(*this == other);
}

template<typename T>
T& Matrix<T>:: operator()(size_t row, size_t col) {
    return m_data[row * m_cols + col];
}

template<typename T>
const T& Matrix<T>:: operator()(size_t row, size_t col) const {
    return m_data[row * m_cols + col];
}

template<typename T>
void Matrix<T>:: printNonZeroElements() const {
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            if (m_data[i * m_cols + j] != T{}) {
                std::cout << "Element at (" << i << ", " << j << ") is " << m_data[i * m_cols + j] << std::endl;
            }
        }
    }
}

template<typename T>
size_t Matrix<T>:: getRows() const { return m_rows; }

template<typename T>
size_t Matrix<T>:: getCols() const { return m_cols; }

template<typename T>
const std::vector<T>& Matrix<T>:: getData() const { return m_data; }

template<typename T>
const std::vector<int>& Matrix<T>:: getP() const { return pMat; }

template<typename T>
void Matrix<T>:: setP(std::vector<int> p) {
    pMat = std::move(p);
}

template<typename T>
Matrix<T> Matrix<T>:: transpose() const {
    Matrix<T> result(m_cols, m_rows);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>:: multiplyParallel(const Matrix<T>& other) const {
    if (m_cols != other.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(m_rows, other.getCols());
    std::vector<int> index(m_rows);
    std::iota(index.begin(), index.end(), 0);
    std::for_each(std::execution::par, index.begin(), index.end(), [&other, this,result](int i) {
        for (size_t j = 0; j < other.getCols(); ++j) {
            for (size_t k = 0; k < m_cols; ++k) {
                result(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    });

    return result;
}

template<typename T>
Matrix<T> Matrix<T>:: blockMultiplication(const Matrix<T>& other, int blockSize) const {
    if (m_cols != other.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    if (blockSize <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Number of concurrent threads supported: " << n << std::endl;

    Matrix<T> result(m_rows, other.getCols());
    for (size_t i = 0; i < m_rows; i += blockSize) {
        for (size_t j = 0; j < other.getCols(); j += blockSize) {
            for (size_t k = 0; k < m_cols; k += blockSize) {
                for (size_t ii = i; ii < std::min(i + blockSize, m_rows); ++ii) {
                    for (size_t jj = j; jj < std::min(j + blockSize, other.getCols()); ++jj) {
                        for (size_t kk = k; kk < std::min(k + blockSize, m_cols); ++kk) {
                            result(ii, jj) += (*this)(ii, kk) * other(kk, jj);
                        }
                    }
                }
            }
        }
    }

    return result;
}

template<typename T>
Matrix<T> Matrix<T>:: parallelBlockMultiply(const Matrix<T>& other, int blockSize) const {
    if (m_cols == 0 || other.getRows() == 0 || m_cols != other.getRows()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    if (blockSize <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Number of concurrent threads supported: " << n << std::endl;

    int rows1 = m_rows;
    int cols1 = m_cols;
    int cols2 = other.getCols();

    Matrix<T> product(rows1, cols2, T{0});

    std::vector<std::pair<int, int>> blockPairs;
    for (int ii = 0; ii < rows1; ii += blockSize) {
        for (int jj = 0; jj < cols2; jj += blockSize) {
            blockPairs.emplace_back(ii, jj);
        }
    }

    std::for_each(std::execution::par, blockPairs.begin(), blockPairs.end(),
                  [this, rows1, cols1, cols2, blockSize, &other, &product](const std::pair<int, int>& blockPair) {
                      int ii = blockPair.first;
                      int jj = blockPair.second;
                      for (int kk = 0; kk < cols1; kk += blockSize) {
                          for (int i = ii; i < std::min(ii + blockSize, rows1); ++i) {
                              for (int j = jj; j < std::min(jj + blockSize, cols2); ++j) {
                                  for (int k = kk; k < std::min(kk + blockSize, cols1); ++k) {
                                      product(i, j) += (*this)(i, k) * other(k, j);
                                  }
                              }
                          }
                      }
                  });

    return product;
}

template<typename T>
Matrix<T> Matrix<T>::LU_Factorization(int blockLength, int smallBlockSize) const {
    if (this->m_rows != this->m_cols) {
        throw std::invalid_argument("Matrix is not square.");
    }
    Matrix<T> result(this->m_rows, this->m_cols, this->m_data);
    std::vector<int> p(result.getRows());
    std::iota(p.begin(), p.end(), 0);
    // call private function
    DGETRF_intern(result.m_data, result.getRows(), blockLength, p, smallBlockSize);
    result.setP(p);
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::Parallel_LU_Factorization(int blockLength, int smallBlockSize) const {
    if (this->m_rows != this->m_cols) {
        throw std::invalid_argument("Matrix is not square.");
    }
    Matrix<T> result(this->m_rows, this->m_cols, this->m_data);
    std::vector<int> p(result.getRows());
    std::iota(p.begin(), p.end(), 0);
    // call private function
    PDGETRF_intern(result.m_data, result.getRows(), blockLength, p, smallBlockSize);
    result.setP(p);
    return result;
}

//Implementation of the private functions:

template<typename T>
int Matrix<T>::IDAMAX(const std::vector<T>& mat, int blockLength, int BigDimension, int k, int blockStart) const {
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

template<typename T>
void Matrix<T>:: DSWAP(std::vector<T>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) const {
    if (row1 == row2) return;
    for (int i = blockStartCol; i < endCol; ++i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    }
}

template<typename T>
void Matrix<T>:: SwapParray(std::vector<int>& pMat, int row1, int row2) const {
    if (row1 == row2) return;
    std::swap(pMat[row1], pMat[row2]);
}

template<typename T>
void Matrix<T>:: DSCAL(std::vector<T>& mat, int n,int blockLength, int k, int blockStart) const {
    // make k-th column but in a block all divided by the diagonal element
    for (int i = k+1; i < blockStart+blockLength; ++i) {
        mat[i*n + k] /= mat[k*n + k];
    }
}

template<typename T>
void Matrix<T>:: DGER(std::vector<T>& mat, int bigDimension, int k, int blockStart, int blockLength) const {
    for (int i = k+1; i < blockStart+blockLength; ++i) {
        for (int j = k+1; j < blockStart+blockLength; ++j) {
            mat[i*bigDimension + j] -= mat[i*bigDimension + k] * mat[k*bigDimension + j];
        }
    }
}

template<typename T>
void Matrix<T>:: DLASWP(std::vector<T>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) const {
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

template<typename T>
void Matrix<T>:: DTRSM(std::vector<T>& mat, int n, int blockStart, int blockLength) const {
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

template<typename T>
bool Matrix<T>:: DGETF2(std::vector<T>& mat, std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) const {
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

template<typename T>
void Matrix<T>:: DTRSM_L21(std::vector<T>& mat, int n, int blockStart, int blockLength) const {
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

template<typename T>
void Matrix<T>:: BLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength) const {
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

template<typename T>
void Matrix<T>::DGETRF_intern(std::vector<T> &mat, int n, int blockLength, std::vector<int> &pMat, int smallBlockSize) const {
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
        BLOCK_DGEMM(mat, n, blockStart, blockLength, smallBlockSize);
    }
    // if there is a smaller block left
    if (blockStart < n) {
        DGETF2(mat, pMat, n-blockStart, blockStart, n);
        DLASWP(mat, pMat, n, blockStart, n-blockStart);
    }
}

template<typename T>
void Matrix<T>::PDGETRF_intern(std::vector<float> &mat, int n, int blockLength, std::vector<int> &pMat, int smallBlockSize) const {
    int blockStart = 0;
    for (; blockStart+blockLength <= n; blockStart+=blockLength) {

        // applied LU factorization to A11, output L11 and U11 and P11
        PDGETF2(mat, pMat, blockLength, blockStart, n);

        // PDLASWP -  Apply row interchanges to the left and the right of this block.
        PDLASWP(mat, pMat, n, blockStart, blockLength);

        // compute U12
        PDTRSM(mat, n, blockStart, blockLength);

        // compute L21
        // L21 = A21 * U11^-1
        PDTRSM_L21(mat, n, blockStart, blockLength);

        // update A22 here
        // A22 = A22 - L21 * U12
        PBLOCK_DGEMM(mat, n, blockStart, blockLength, smallBlockSize);
    }
    // if there is a smaller block left
    if (blockStart < n) {
        PDGETF2(mat, pMat, n-blockStart, blockStart, n);
        PDLASWP(mat, pMat, n, blockStart, n-blockStart);
    }
}

template<typename T>
void Matrix<T>:: PDTRSM(std::vector<float>& mat, int n, int blockStart, int blockLength) const {
    std::vector<int> indicesJ(n - blockStart - blockLength);
    std::iota(indicesJ.begin(), indicesJ.end(), blockStart + blockLength);
    for (int i = blockStart; i < blockStart + blockLength; ++i) {
        std::for_each(std::execution::par, indicesJ.begin(), indicesJ.end(), [blockStart,i,n,&mat](int j) {
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
template<typename T>
void Matrix<T>:: PDTRSM_L21(std::vector<float>& mat, int n, int blockStart, int blockLength) const {
    std::vector<int> indices(n - blockStart - blockLength);
    std::iota(indices.begin(), indices.end(), blockStart + blockLength);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [blockStart,blockLength,&mat,n](int i) {
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
//template<typename T>
//void Matrix<T>:: PDGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength) const {
//    std::vector<int> indices(n - blockStart - blockLength);
//    std::iota(indices.begin(), indices.end(), blockStart + blockLength);
//
//    // this parallel will not have any problem, because mat[i * n + j] -= sum; will not affect the value of mat in the loop above
//    // because the corresponding area is A22.
//    // sum += mat[i * n + k] * mat[k * n + j]; the value taken here is from L21 and U12
//    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i) {
//        for (int j = blockStart + blockLength; j < n; ++j) {
//            float sum = 0.0;
//            for (int k = blockStart; k < blockStart + blockLength; ++k) {
//                sum += mat[i * n + k] * mat[k * n + j];
//            }
//            mat[i * n + j] -= sum;
//        }
//    });
//}

template<typename T>
void Matrix<T>:: PBLOCK_DGEMM(std::vector<float>& mat, int n, int blockStart, int blockLength, int smallBlockLength) const {
    // can assume that smallBlockLength is equal to blockLength, which may be tested later
    int A22start = blockStart + blockLength;

    // Create a vector to hold all the block pairs for parallel processing
    std::vector<std::pair<int, int>> blockPairs;
    for (int ii = A22start; ii < n; ii += smallBlockLength) {
        for (int jj = A22start; jj < n; jj += smallBlockLength) {
            blockPairs.emplace_back(ii, jj);
        }
    }

    // Use parallel algorithm to process each block
    std::for_each(std::execution::par, blockPairs.begin(), blockPairs.end(),
                  [blockStart,A22start,smallBlockLength,n,&mat](const std::pair<int, int>& block) {
                      int ii = block.first;
                      int jj = block.second;
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
                  });
}

// PA = LU
// Naive LU decomposition, calculate the L and U for a block in mat matrix
template<typename T>
bool Matrix<T>:: PDGETF2(std::vector<float>& mat,std::vector<int>& pMat, int blockLength, int blockStartCol, int bigDimension) const {
    for (int k = blockStartCol; k < blockStartCol+blockLength; ++k) {
        // find the pivot value in the k-th column and set that row index to pivotRow
        int pivotRow = IDAMAX(mat, blockLength, bigDimension,k, blockStartCol);
        // swap the pivot row with the kth row
        PDSWAP(mat, blockStartCol, blockStartCol+blockLength, k, pivotRow, bigDimension);
        SwapParray(pMat, k, pivotRow);

        // these two functions can be combined into one function! load once!
        PDSCAL(mat, bigDimension, blockLength, k,blockStartCol);
        PDGER(mat, bigDimension, k, blockStartCol, blockLength);
    }
    return true;
}

template<typename T>
void Matrix<T>:: PDSCAL(std::vector<float>& mat, int n,int blockLength, int k, int blockStart) const {
    int numElements = blockStart + blockLength - (k + 1);
    std::vector<int> indices(numElements);
    std::iota(indices.begin(), indices.end(), k + 1);
    float diagonalElement = mat[k * n + k];

    // every thread share the same diagonalElement
    std::for_each(std::execution::par, indices.begin(), indices.end(), [n,k,diagonalElement,&mat](int i) {
        mat[i * n + k] /= diagonalElement;
    });
}

// perform a rank-k update on a matrix but only within a given range of rows and columns
template<typename T>
void Matrix<T>:: PDGER(std::vector<float>& mat, int bigDimension, int k, int blockStart, int blockLength) const {
    std::vector<int> indices(blockStart+blockLength - (k + 1));
    std::iota(indices.begin(), indices.end(), k + 1);

    std::for_each(std::execution::par, indices.begin(), indices.end(), [k,blockStart,blockLength,&mat,bigDimension](int i) {
        for (int j = k + 1; j < blockStart + blockLength; ++j) {
            mat[i * bigDimension + j] -= mat[i * bigDimension + k] * mat[k * bigDimension + j];
        }
    });
}

// Apply row interchanges to the left and the right of the panel.
template<typename T>
void Matrix<T>:: PDLASWP(std::vector<float>& mat, std::vector<int>& pMat, int n, int blockStart, int blockLength) const {
    // create a vector to keep track of which rows have been swapped to avoid duplicate swaps

    // create a vector A, the values are from blockStart to blockStart+blockLength-1
    std::vector<int> A(blockLength);
    std::iota(A.begin(), A.end(), 0);

    // no need to parallelize this loop, A is just a small vector with blockLength elements
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

            // swap the left part of the matrix

            // these two swaps are parallel, because the size of the part that needs to be swapped is n-blockLength,
            // which is large, so it needs to be processed in parallel
            PDSWAP(mat, 0, blockStart, i+blockStart, swapIndex+blockStart, n);
            // swap the right part of the matrix
            PDSWAP(mat, blockStart+blockLength, n, i+blockStart, swapIndex+blockStart, n);

            std::swap(A[i], A[swapIndex]);
        }
    }
}

template<typename T>
void Matrix<T>:: PDSWAP(std::vector<float>& mat, int blockStartCol, int endCol, int row1, int row2, int bigDimension) const {
    if (row1 == row2) return;
    std::vector<int> indices(endCol - blockStartCol);
    std::iota(indices.begin(), indices.end(), blockStartCol);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&mat,row1,bigDimension,row2](int i) {
        std::swap(mat[row1*bigDimension + i], mat[row2*bigDimension + i]);
    });
}

#endif //CPP_MATRIX_MULTI_GENERALMATRIXCLASS_H
