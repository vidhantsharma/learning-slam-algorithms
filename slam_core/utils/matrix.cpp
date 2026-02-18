#include "matrix.h"

#include <cmath>

namespace kf {

Matrix::Matrix() : rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument("Matrix data size mismatch");
    }
}

double& Matrix::operator()(size_t r, size_t c) {
    return data_.at(r * cols_ + c);
}

double Matrix::operator()(size_t r, size_t c) const {
    return data_.at(r * cols_ + c);
}

void Matrix::assert_same_shape(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix shape mismatch");
    }
}

Matrix Matrix::identity(size_t size) {
    Matrix m(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        m(i, i) = 1.0;
    }
    return m;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_, 0.0);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < cols_; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::inverse() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square to invert");
    }

    size_t n = rows_;
    Matrix aug(n, 2 * n, 0.0);
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            aug(r, c) = (*this)(r, c);
        }
        aug(r, n + r) = 1.0;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        double max_val = std::abs(aug(i, i));
        for (size_t r = i + 1; r < n; ++r) {
            double val = std::abs(aug(r, i));
            if (val > max_val) {
                max_val = val;
                pivot = r;
            }
        }

        if (std::abs(aug(pivot, i)) < 1e-12) {
            throw std::runtime_error("Matrix is singular");
        }

        if (pivot != i) {
            for (size_t c = 0; c < 2 * n; ++c) {
                std::swap(aug(i, c), aug(pivot, c));
            }
        }

        double diag = aug(i, i);
        for (size_t c = 0; c < 2 * n; ++c) {
            aug(i, c) /= diag;
        }

        for (size_t r = 0; r < n; ++r) {
            if (r == i) {
                continue;
            }
            double factor = aug(r, i);
            for (size_t c = 0; c < 2 * n; ++c) {
                aug(r, c) -= factor * aug(i, c);
            }
        }
    }

    Matrix inv(n, n, 0.0);
    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < n; ++c) {
            inv(r, c) = aug(r, n + c);
        }
    }

    return inv;
}

Matrix Matrix::operator+(const Matrix& other) const {
    assert_same_shape(other);
    Matrix result(rows_, cols_, 0.0);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    assert_same_shape(other);
    Matrix result(rows_, cols_, 0.0);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix multiplication shape mismatch");
    }
    Matrix result(rows_, other.cols_, 0.0);
    for (size_t r = 0; r < rows_; ++r) {
        for (size_t c = 0; c < other.cols_; ++c) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += (*this)(r, k) * other(k, c);
            }
            result(r, c) = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_, 0.0);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    assert_same_shape(other);
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    assert_same_shape(other);
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

} // namespace kf
