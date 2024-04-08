#include "IMatrix.h"

// Sparse matrix using Compressed Sparse Row (CSR) format
template<typename T>
class SparseMatrixCSR : public IMatrix<T> {
private:
    std::vector<T> values;
    std::vector<size_t> colIndices;
    std::vector<size_t> rowPointers;
    size_t rows, cols;

public:
    SparseMatrixCSR(size_t numRows, size_t numCols);
    SparseMatrixCSR(const std::vector<std::vector<T>>& denseMatrix);

    SparseMatrixCSR<T>& operator=(const SparseMatrixCSR<T>& other);

    void printNonZeroElements() const override;
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;
    size_t getRows() const override;
    size_t getCols() const override;

    friend SparseMatrixCSR<T> operator+(const SparseMatrixCSR<T>& a, const SparseMatrixCSR<T>& b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw std::invalid_argument("Matrix dimensions do not match.");
        }

        SparseMatrixCSR<T> result(a.rows, a.cols);

        for (size_t i = 0; i < a.rows; ++i) {
            size_t aRowStart = a.rowPointers[i];
            size_t aRowEnd = a.rowPointers[i + 1];
            size_t bRowStart = b.rowPointers[i];
            size_t bRowEnd = b.rowPointers[i + 1];

            size_t aPos = aRowStart, bPos = bRowStart;

            while (aPos < aRowEnd && bPos < bRowEnd) {
                if (a.colIndices[aPos] == b.colIndices[bPos]) {
                    result.values.push_back(a.values[aPos] + b.values[bPos]);
                    result.colIndices.push_back(a.colIndices[aPos]);
                    aPos++;
                    bPos++;
                } else if (a.colIndices[aPos] < b.colIndices[bPos]) {
                    result.values.push_back(a.values[aPos]);
                    result.colIndices.push_back(a.colIndices[aPos]);
                    aPos++;
                } else {
                    result.values.push_back(b.values[bPos]);
                    result.colIndices.push_back(b.colIndices[bPos]);
                    bPos++;
                }
            }

            while (aPos < aRowEnd) {
                result.values.push_back(a.values[aPos]);
                result.colIndices.push_back(a.colIndices[aPos]);
                aPos++;
            }

            while (bPos < bRowEnd) {
                result.values.push_back(b.values[bPos]);
                result.colIndices.push_back(b.colIndices[bPos]);
                bPos++;
            }

            result.rowPointers.push_back(result.values.size());
        }

        return result;
    }
};

template<typename T>
SparseMatrixCSR<T>& SparseMatrixCSR<T>:: operator=(const SparseMatrixCSR<T>& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        values = other.values;
        colIndices = other.colIndices;
        rowPointers = other.rowPointers;
    }
    return *this;
}

template<typename T>
SparseMatrixCSR<T>:: SparseMatrixCSR(const std::vector<std::vector<T>>& denseMatrix) {
    rows = denseMatrix.size();
    cols = rows ? denseMatrix[0].size() : 0;

    rowPointers.push_back(0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (denseMatrix[i][j] != T{}) {
                values.push_back(denseMatrix[i][j]);
                colIndices.push_back(j);
            }
        }
        rowPointers.push_back(values.size());
    }
}

template<typename T>
SparseMatrixCSR<T>::SparseMatrixCSR(size_t numRows, size_t numCols){
    rows = numRows;
    cols = numCols;
    values.clear();
    colIndices.clear();
    rowPointers.clear();
    rowPointers.push_back(0);
}

template<typename T>
void SparseMatrixCSR<T>:: printNonZeroElements() const {
    std::cout << "Non-zero elements and their positions:" << std::endl;
    for (size_t i = 0; i < rows; ++i) {
        size_t start = rowPointers[i];
        size_t end = rowPointers[i + 1];
        for (size_t j = start; j < end; ++j) {
            T value = values[j];
            size_t col = colIndices[j];
            std::cout << "Value: " << value << " at (" << i << ", " << col << ")" << std::endl;
        }
    }
}

template<typename T>
T& SparseMatrixCSR<T>:: operator()(size_t row, size_t col){
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    size_t start = rowPointers[row];
    size_t end = rowPointers[row + 1];

    for (size_t i = start; i < end; i++) {
        if (colIndices[i] == col) {
            return values[i];
        }
    }
    throw std::logic_error("Element not found");
}

template<typename T>
const T& SparseMatrixCSR<T>:: operator()(size_t row, size_t col) const{
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    size_t start = rowPointers[row];
    size_t end = rowPointers[row + 1];

    for (size_t i = start; i < end; i++) {
        if (colIndices[i] == col) {
            return values[i];
        }
    }
    throw std::logic_error("Element not found");
}

template<typename T>
size_t SparseMatrixCSR<T>:: getRows() const{
    return rows;
}

template<typename T>
size_t SparseMatrixCSR<T>:: getCols() const{
    return cols;
}

