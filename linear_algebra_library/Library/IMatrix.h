#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <execution>
#include <algorithm>
#include <iomanip>
#include <utility>
#include <random>
#include <thread>

#ifndef CPP_MATRIX_MULTI_ABSTRACTBASEMATRIX_H
#define CPP_MATRIX_MULTI_ABSTRACTBASEMATRIX_H

template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
// this is an abstract class
class IMatrix {
public:
    virtual ~IMatrix() {}
    // English : = 0: This indicates that the function is a pure virtual function (Pure Virtual Function), used to declare an interface.
    // In the base class, the pure virtual function does not have a specific implementation, but requires the derived class to provide an implementation.
    // This concept is used to implement abstract classes. A class containing pure virtual functions cannot be instantiated and must be inherited.
    virtual T& operator()(size_t row, size_t col) = 0;
    virtual const T& operator()(size_t row, size_t col) const = 0;
    virtual size_t getRows() const = 0;
    virtual size_t getCols() const = 0;
    virtual void printNonZeroElements() const = 0;
};

#endif //CPP_MATRIX_MULTI_ABSTRACTBASEMATRIX_H

