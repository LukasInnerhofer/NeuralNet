#ifndef VECTOR_H
#define VECTOR_H

//#include <vector>
#include <cereal/types/vector.hpp>

template<typename T>
class Vector : public std::vector<T>
{
public:
	Vector() : std::vector<T>() {}
	Vector(const std::vector<T> &vector) : std::vector<T>(vector) {}
	Vector(const std::initializer_list<T> &list) : std::vector<T>(list) {}
	Vector(const unsigned int &n, const T element) : std::vector<T>(n, element) {}

	Vector<T> operator-(const Vector<T> &vector)
	{
		try
		{
			if (this->size() != vector.size())
			{
				throw std::invalid_argument("Vector subtraction requires two vectors of the same size");
			}
		}
		catch (const std::invalid_argument &e)
		{
			std::cerr << "Invalid argument. " << e.what() << std::endl;
			return Vector<T>();
		}

		Vector<T> difference;
		for (decltype(this->size()) it = 0; it < this->size(); ++it)
		{
			difference.push_back((*this)[it] - vector[it]);
		}
		return difference;
	}

	Vector<T> operator+(const Vector<T> &vector)
	{
		try
		{
			if (this->size() != vector.size())
			{
				throw std::invalid_argument("Vector addition requires two vectors of the same size");
			}
		}
		catch (const std::invalid_argument &e)
		{
			std::cerr << "Invalid argument. " << e.what() << std::endl;
			return Vector<T>();
		}

		Vector<T> sum;
		for (decltype(this->size()) it = 0; it < this->size(); ++it)
		{
			sum.push_back((*this)[it] + vector[it]);
		}
		return sum;
	}

	Vector<T> hadamard(const Vector<T> &vector)
	{
		try
		{
			if (this->size() != vector.size())
			{
				throw std::invalid_argument("Hadamard requires two vectors of the same size.");
			}
		}
		catch (const std::invalid_argument &e)
		{
			std::cerr << "Invalid argument. " << e.what() << std::endl;
			return Vector<T>();
		}

		Vector<T> hadamard;
		for (decltype(this->size()) it = 0; it < this->size(); ++it)
		{
			hadamard.push_back((*this)[it] * vector[it]);
		}

		return hadamard;
	}
};

#endif // VECTOR_H