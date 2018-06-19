#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

template<typename T>
class Matrix : public Vector<Vector<T>>
{
public:
	using Vector<Vector<T>>::Vector;

	Matrix() : Vector<Vector<T>>() {}

	Matrix<T> transpose() const
	{
		auto transposedMatrix = Matrix<T>((*this)[0].size(), Vector<T>());
		for (unsigned int itRows = 0; itRows < (*this)[0].size(); ++itRows)
		{
			for (unsigned int itColumns = 0; itColumns < this->size(); ++itColumns)
			{
				transposedMatrix[itRows].push_back(0);
				transposedMatrix[itRows][itColumns] = (*this)[itColumns][itRows];
			}
		}
		return transposedMatrix;
	}

	Vector<T> operator*(const Vector<T> &vector)
	{
		try
		{
			if (this->size() != vector.size())
			{
				throw std::invalid_argument("Number of matrix columns must be equal to the number of vector rows.");
			}
		}
		catch (const std::invalid_argument &e)
		{
			std::cerr << "Invalid argument. " << e.what() << std::endl;
			return Vector<T>();
		}

		Vector<T> returnVector;

		for (decltype((*this)[0].size()) itRows = 0; itRows < (*this)[0].size(); ++itRows)
		{
			returnVector.push_back(0);
			for (decltype(this->size()) itColumns = 0; itColumns < this->size(); ++itColumns)
			{
				returnVector[itRows] += (*this)[itColumns][itRows] * vector[itColumns];
			}
		}

		return returnVector;
	}
};

#endif // MATRIX_H