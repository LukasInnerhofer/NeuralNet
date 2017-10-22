#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
std::vector<T> operator-(const std::vector<T> &vector1, const std::vector<T> &vector2);

namespace matrixMath
{
	template<typename T>
	std::vector<T> hadamard(const std::vector<T> &vector1, const std::vector<T> &vector2);
}

#endif // MATRIX_MATH_H