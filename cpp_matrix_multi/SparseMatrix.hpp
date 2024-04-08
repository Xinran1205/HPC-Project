#include "IMatrix.h"

// use three vectors to store the non-zero elements and their positions
template<typename T>
class SparseMatrix : public IMatrix<T> {
private:
    std::vector<T> values;
    std::vector<size_t> rowIndices;
    std::vector<size_t> colIndices;
    size_t rows, cols;

public:
    SparseMatrix(size_t numRows, size_t numCols);
    SparseMatrix(std::initializer_list<std::initializer_list<T>> matrix);
    SparseMatrix(const std::vector<std::vector<T>>& data);

    SparseMatrix<T> transpose() const;
    void setElement(size_t row, size_t col, T value);
    void printNonZeroElements() const override;
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;
    size_t getRows() const override;
    size_t getCols() const override;

    friend SparseMatrix<T> operator+(const SparseMatrix<T>& a, const SparseMatrix<T>& b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            throw std::invalid_argument("Matrices dimensions do not match");
        }

        SparseMatrix<T> result(a.rows, a.cols);

        // put the elements of the first matrix into the result matrix
        for (size_t i = 0; i < a.values.size(); i++) {
            result.setElement(a.rowIndices[i], a.colIndices[i], a.values[i]);
        }

        // add the elements of the second matrix to the result matrix, if the position is the same then add them
        for (size_t i = 0; i < b.values.size(); i++) {
            size_t row = b.rowIndices[i];
            size_t col = b.colIndices[i];
            T value = b.values[i];

            // if the result matrix already has a value at this position, then add them
            for (size_t j = 0; j < result.values.size(); j++) {
                if (result.rowIndices[j] == row && result.colIndices[j] == col) {
                    result.values[j] += value;
                    goto next;
                }
            }

            // if the result matrix does not have a value at this position, then set it directly
            result.setElement(row, col, value);

            next:;
        }

        return result;
    }
};

template<typename T>
SparseMatrix<T>::SparseMatrix(size_t numRows, size_t numCols) : rows(numRows), cols(numCols) {}

template<typename T>
SparseMatrix<T>::SparseMatrix(std::initializer_list<std::initializer_list<T>> matrix) {
    rows = matrix.size();
    cols = rows ? matrix.begin()->size() : 0;
    for (size_t i = 0; i < rows; ++i) {
        auto row = *(matrix.begin() + i);
        for (size_t j = 0; j < row.size(); ++j) {
            T value = *(row.begin() + j);
            if (value != T{}) {
                values.push_back(value);
                rowIndices.push_back(i);
                colIndices.push_back(j);
            }
        }
    }
}

template<typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<std::vector<T>>& data) {
    rows = data.size();
    cols = rows ? data[0].size() : 0;

    for (size_t i = 0; i < rows; ++i) {
        const auto& row = data[i];
        for (size_t j = 0; j < row.size(); ++j) {
            T value = row[j];
            if (value != T{}) {
                values.push_back(value);
                rowIndices.push_back(i);
                colIndices.push_back(j);
            }
        }
    }
}

template<typename T>
SparseMatrix<T> SparseMatrix<T>:: transpose() const {
    SparseMatrix<T> result(cols, rows);
    for (size_t i = 0; i < values.size(); ++i) {
        result.setElement(colIndices[i], rowIndices[i], values[i]);
    }
    return result;
}

template<typename T>
void SparseMatrix<T>:: printNonZeroElements() const {
    std::cout << "Non-zero elements and their positions:" << std::endl;
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << "Value: " << values[i]
                  << " at (" << rowIndices[i]
                  << ", " << colIndices[i] << ")" << std::endl;
    }
}

template<typename T>
void SparseMatrix<T>:: setElement(size_t row, size_t col, T value) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (rowIndices[i] == row && colIndices[i] == col) {
            values[i] = value;
            return;
        }
    }

    values.push_back(value);
    rowIndices.push_back(row);
    colIndices.push_back(col);
}

template<typename T>
T& SparseMatrix<T>:: operator()(size_t row, size_t col){
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of range");
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (rowIndices[i] == row && colIndices[i] == col) {
            return values[i];
        }
    }
    throw std::logic_error("Element not found");
}

template<typename T>
const T& SparseMatrix<T>:: operator()(size_t row, size_t col) const{
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Index out of range");
    }

    for (size_t i = 0; i < values.size(); ++i) {
        if (rowIndices[i] == row && colIndices[i] == col) {
            return values[i];
        }
    }
    throw std::logic_error("Element not found");
}

template<typename T>
size_t SparseMatrix<T>:: getRows() const{
    return rows;
}

template<typename T>
size_t SparseMatrix<T>:: getCols() const{
    return cols;
}