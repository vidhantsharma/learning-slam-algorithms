#pragma once

#include <vector>
#include <stdexcept>
#include <initializer_list>

namespace kf {

class Matrix {
public:
    Matrix();
    Matrix(size_t rows, size_t cols, double value = 0.0);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    double& operator()(size_t r, size_t c);
    double operator()(size_t r, size_t c) const;

    std::vector<double> data() const { return data_; }

    static Matrix identity(size_t size);
    static Matrix zeros(size_t rows, size_t cols);

    Matrix transpose() const;
    Matrix inverse() const;
    double det() const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

private:
    size_t rows_{};
    size_t cols_{};
    std::vector<double> data_{};

    void assert_same_shape(const Matrix& other) const;
};

Matrix operator*(double scalar, const Matrix& m);

} // namespace kf
