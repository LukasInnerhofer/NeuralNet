#include <vector>
#include <iostream>
#include <stdexcept>

template<typename T>
std::vector<T> operator-(const std::vector<T> &vector1, const std::vector<T> &vector2)
{
	try
	{
		if (vector1.size() != vector2.size())
		{
			throw std::invalid_argument("Matrix subtraction requires two matrices of the same size");
		}
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "Invalid argument. " << e.what() << std::endl;
		return std::vector<T>();
	}

	std::vector<T> difference;
	for (unsigned int it = 0; it < vector1.size(); ++it)
	{
		difference.push_back(vector1[it] - vector2[it]);
	}
	return difference;
}

namespace matrixMath
{
	template<typename T>
	std::vector<T> hadamard(const std::vector<T> &vector1, const std::vector<T> &vector2)
	{
		try
		{
			if (vector1.size() != vector2.size())
			{
				throw std::invalid_argument("Hadamard requires two matrices of the same size.");
			}
		}
		catch (const std::invalid_argument &e)
		{
			std::cerr << "Invalid argument. " << e.what() << std::endl;
			return std::vector<T>();
		}

		std::vector<T> hadamard;
		for (unsigned int it = 0; it < vector1.size(); ++it)
		{
			hadamard.push_back(vector1[it] * vector2[it]);
		}

		return hadamard;
	}

	template<typename T>
	std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &matrix)
	{
		auto transposedMatrix = std::vector<std::vector<T>>(matrix[0].size(), std::vector<T>());
		for (unsigned int itRows = 0; itRows < matrix[0].size(); ++itRows)
		{
			for (unsigned int itColumns = 0; itColumns < matrix.size(); ++itColumns)
			{
				transposedMatrix[itRows].push_back(0);
				if (itColumns == itRows)
				{
					transposedMatrix[itColumns][itRows] = matrix[itColumns][itRows];
				}
				else
				{
					transposedMatrix[itRows][itColumns] = matrix[itColumns][itRows];
				}
			}
		}
		return transposedMatrix;
	}
}