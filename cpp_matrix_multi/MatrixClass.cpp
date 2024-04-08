#include <vector>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>
#include "IMatrix.h"

//is_arithmetic ensure that the type is a numeric type
template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
class Matrix{
public:

    // 默认构造函数，这个语法叫做成员初始化列表
    Matrix() : m_rows(0), m_cols(0) {}

    // 从行数和列数创建矩阵，默认元素为0
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(rows * cols, T()) {}

    // 从行数、列数和一个初始值创建矩阵
    Matrix(size_t rows, size_t cols, T fillValue) : m_rows(rows), m_cols(cols), m_data(rows * cols, fillValue) {}

    // 从二维向量创建矩阵
    Matrix(const std::vector<std::vector<T>>& data) : m_rows(data.size()), m_cols(data.empty() ? 0 : data[0].size()) {
        for (const auto& row : data) {
            if (row.size() != m_cols) throw std::invalid_argument("Irregular matrix shape");
            m_data.insert(m_data.end(), row.begin(), row.end());
        }
    }

    // 使用列表初始化创建矩阵
    Matrix(std::initializer_list<std::initializer_list<T>> list) : m_rows(list.size()), m_cols(list.begin()->size()) {
        for (const auto& row : list) {
            if (row.size() != m_cols) throw std::invalid_argument("Irregular matrix shape");
            m_data.insert(m_data.end(), row.begin(), row.end());
        }
    }

    // 重载赋值运算符，默认浅拷贝，拷贝值，但是在遇到需要释放内存的时候会出问题
    // 以下就是浅拷贝的代码，写不写无所谓
    // 使用方法 例如Matrix<int> a;
    // 他会自动把后者根据构造函数转换成一个Matrix<T>&
    //  a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    Matrix<T>& operator=(const Matrix<T>& other) {
//        if (this != &other) {
//            m_rows = other.m_rows;
//            m_cols = other.m_cols;
//            m_data = other.m_data;
//        }
//        return *this;
//    }

    // 重载函数调用运算符！ 用来访问元素（非const和const版本）
    // 用于更便捷的访问矩阵元素，可以通过m(i, j)的方式访问矩阵元素
    T& operator()(size_t row, size_t col) {
        return m_data[row * m_cols + col];
    }
    const T& operator()(size_t row, size_t col) const {
        return m_data[row * m_cols + col];
    }

    // 获取行数和列数
    size_t getRows() const { return m_rows; }
    size_t getCols() const { return m_cols; }

    // 获取矩阵数据
    // 返回一个const引用，避免数据被修改
    const std::vector<T>& getData() const { return m_data; }

private:
    size_t m_rows, m_cols;
    std::vector<T> m_data; // 使用一维数组存储矩阵数据
};

// 重载cout，用于输出矩阵和矩阵大小
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) {
    for (size_t i = 0; i < mat.getRows(); ++i) {
        for (size_t j = 0; j < mat.getCols(); ++j) {
            os << std::setw(1) << mat(i, j) << " ";
        }
        os << std::endl;
    }

    os << "Matrix size: " << mat.getRows() << "x" << mat.getCols() << std::endl;
    return os;
}
// 重载加法
template<typename T>
Matrix<T> operator+(const Matrix<T>& mat1, const Matrix<T>& mat2) {
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

// 重载减法
template<typename T>
Matrix<T> operator-(const Matrix<T>& mat1, const Matrix<T>& mat2) {
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


// overload the multiplication operator
// naive matrix multiplication
template<typename T>
Matrix<T> operator*(const Matrix<T>& mat1, const Matrix<T>& mat2) {
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

// parallel naive matrix multiplication
template<typename T>
Matrix<T> multiplyParallel(const Matrix<T>& mat1, const Matrix<T>& mat2) {
    if (mat1.getCols() != mat2.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix<T> result(mat1.getRows(), mat2.getCols());
    std::vector<int> index(mat1.getRows());
    std::iota(index.begin(), index.end(), 0);
    std::for_each(std::execution::par, index.begin(), index.end(), [&](int i) {
        for (size_t j = 0; j < mat2.getCols(); ++j) {
            for (size_t k = 0; k < mat1.getCols(); ++k) {
                result(i, j) += mat1(i, k) * mat2(k, j);
            }
        }
    });

    return result;
}

// block matrix multiplication // default block size is 2
// Correct!
template<typename T>
Matrix<T> blockMultiplication(const Matrix<T>& mat1, const Matrix<T>& mat2, int blockSize = 2) {
    if (mat1.getCols() != mat2.getRows()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    if (blockSize <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    Matrix<T> result(mat1.getRows(), mat2.getCols());
    for (size_t i = 0; i < mat1.getRows(); i += blockSize) {
        for (size_t j = 0; j < mat2.getCols(); j += blockSize) {
            for (size_t k = 0; k < mat1.getCols(); k += blockSize) {
                for (size_t ii = i; ii < std::min(i + blockSize, mat1.getRows()); ++ii) {
                    for (size_t jj = j; jj < std::min(j + blockSize, mat2.getCols()); ++jj) {
                        for (size_t kk = k; kk < std::min(k + blockSize, mat1.getCols()); ++kk) {
                            result(ii, jj) += mat1(ii, kk) * mat2(kk, jj);
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Parallel block matrix multiplication
// Correct!
template<typename T>
Matrix<T> parallelBlockMultiply(const Matrix<T>& mat1, const Matrix<T>& mat2, int blockSize = 2) {
    if (mat1.getCols() == 0 || mat2.getRows() == 0 || mat1.getCols() != mat2.getRows()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    if (blockSize <= 0) {
        throw std::invalid_argument("Block size must be positive");
    }

    int rows1 = mat1.getRows();
    int cols2 = mat2.getCols();

    Matrix<T> product(rows1, cols2, T{0});

    // calculate the number of blocks in both dimensions
    int blockRows = (rows1 + blockSize - 1) / blockSize;
    int blockCols = (cols2 + blockSize - 1) / blockSize;

    // Perform block matrix multiplication in parallel for each small block
    std::vector<std::pair<int, int>> blocks;
    for (int i = 0; i < blockRows; ++i) {
        for (int j = 0; j < blockCols; ++j) {
            blocks.emplace_back(i, j);
        }
    }

    std::for_each(std::execution::par, blocks.begin(), blocks.end(),
                  [&](std::pair<int, int> block) {
                      int blockRowStart = block.first * blockSize;
                      int blockColStart = block.second * blockSize;
                      for (int ii = blockRowStart; ii < std::min(blockRowStart + blockSize, rows1); ++ii) {
                          for (int jj = blockColStart; jj < std::min(blockColStart + blockSize, cols2); ++jj) {
                              for (int k = 0; k < mat1.getCols(); ++k) {
                                  product(ii, jj) += mat1(ii, k) * mat2(k, jj);
                              }
                          }
                      }
                  });

    return product;
}

//用于测试，后续删除！
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

int main() {
//    // 使用默认构造函数
//    Matrix<int> m1;
//
//    // 从行数和列数创建矩阵
//    Matrix<int> m2(2, 3);
//    std::cout << m2.getRows() << " " << m2.getCols() << std::endl;
//    std::cout << m2.getData().size() << std::endl;
//
//    // 从行数、列数和一个初始值创建矩阵
//    Matrix<int> m3(3, 3, 1);
//    std::cout << m3.getRows() << " " << m3.getCols() << std::endl;
//    std::cout << m3.getData().size() << std::endl;


    // 从二维向量创建矩阵
    std::vector<std::vector<int>> vec{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix<int> m4(vec);

    std ::cout << m4 << std::endl;
//
//
    // 使用列表初始化创建矩阵
    Matrix<int> m5(m4);

//
//
//    // 分块算法乘法
//    Matrix<int> m10 = blockMultiplication(m5, m4, 2);
//    for (int i : m10.getData()) {
//        std::cout << i << " ";
//    }
//
//    // 并行分块算法乘法
//    Matrix<int> m11 = parallelBlockMultiply(m5, m4, 2);
//    for (int i : m11.getData()) {
//        std::cout << i << " ";
//    }

// 创建多个随机矩阵，循环测试
//    std::vector<int> sizes = {6,1,2,3,234,232,231,230,100,4,5,111,101};
//    for (int n : sizes) {
//        auto mat1 = generateRandomMatrix(n, n);
//        auto mat2 = generateRandomMatrix(n, n);
//
//        Matrix<float> m1(mat1);
//        Matrix<float> m2(mat2);
//
//        // 块的大小是1-20随机的数字
//        int blockSize = 1 + rand() % 20;
//
//        // 检测并行分块算法乘法是否正确
//        auto blockProduct = blockMultiplication(m1, m2, blockSize);
//        auto paraBlockProduct = parallelBlockMultiply(m1, m2, blockSize);
//        if (blockProduct.getData() != paraBlockProduct.getData()) {
//            std::cerr << "Block matrix multiplication is incorrect!" << std::endl;
//        }else{
//            std::cout << "correct!" << std::endl;
//        }
//
//    }



    return 0;
}