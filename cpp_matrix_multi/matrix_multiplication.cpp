//
// Created by dell on 2024/2/6.
//

#include "matrix_multiplication.h"

#include <iostream>
#include <vector>

std::vector<std::vector<int>> matrixMultiply(const std::vector<std::vector<int>>& mat1, const std::vector<std::vector<int>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1[0].size() != mat2.size()) {
        throw std::invalid_argument("Matrices cannot be multiplied due to size mismatch");
    }

    int rows1 = mat1.size(), cols1 = mat1[0].size(), cols2 = mat2[0].size();
    std::vector<std::vector<int>> product(rows1, std::vector<int>(cols2, 0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                product[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return product;
}

int main() {
    // Example matrices
    std::vector<std::vector<int>> mat1 = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> mat2 = {{7, 8}, {9, 10}, {11, 12}};

    try {
        auto result = matrixMultiply(mat1, mat2);

        std::cout << "Product of matrices:\n";
        for (const auto& row : result) {
            for (int elem : row) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
