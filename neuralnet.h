#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <cmath>
#include <random>
#include <iostream>
#include <functional>

#include "vector.h"
#include "matrix.h"

class NeuralNet
{
public:
	typedef double (*AFunction)(const double &);

private:
	Matrix<double> inputs;
	Matrix<double> outputs;
	Matrix<double> biases;
	Matrix<AFunction> aFunctions;
	Matrix<AFunction> aFunctionPrimes;
	Vector<Matrix<double>> synapses;

	static Vector<AFunction> availableAFunctions;
	static Vector<AFunction> availableAFunctionPrimes;

	Vector<double> vectorFunction(Vector<AFunction> functions, const Vector<double> &args);

	int randInt(const int &lowerBound, const int &upperBound);
	double randDouble(const double &lowerBound, const double &upperBound);

public:
	Vector<double> getOutputs();
	Matrix<double> getNeurons() const { return outputs; }
	Vector<Matrix<double>> getSynapses() const { return synapses; }

	NeuralNet();
	NeuralNet(const Vector<unsigned int> &neurons, const Vector<Vector<unsigned int>> &aFunctions);

	enum aFunction { Identity, Logistic, TanH, AFunctionMax = TanH };

	void forward(const Vector<double> &inputs);
	void train(const Vector<double> &inputs, const Vector<double> &outputs);

	void mutate();
	NeuralNet breed(const NeuralNet &partner);

	template<class Archive>
	void save(Archive &ar) const
	{
		auto aFunctionIds = Matrix<unsigned int>();
		for (const Vector<AFunction> &vector : aFunctions)
		{
			aFunctionIds.push_back(Vector<unsigned int>());
			for (const AFunction &function : vector)
			{
				const unsigned int id = std::find(availableAFunctions.begin(), availableAFunctions.end(), function) - availableAFunctions.begin();
				aFunctionIds[aFunctionIds.size() - 1].push_back(id);
			}
		}

		ar(CEREAL_NVP(inputs), CEREAL_NVP(outputs), CEREAL_NVP(biases), cereal::make_nvp("activationFunctions", aFunctionIds), CEREAL_NVP(synapses));
	}

	template<class Archive>
	void load(Archive &ar)
	{
		inputs = outputs = biases = Matrix<double>();
		synapses = Vector<Matrix<double>>();
		aFunctions = aFunctionPrimes = Matrix<AFunction>();

		auto aFunctionIds = Matrix<unsigned int>();
		ar(inputs, outputs, biases, aFunctionIds, synapses);

		for (const Vector<unsigned int> &vector : aFunctionIds)
		{
			aFunctions.push_back(Vector<AFunction>());
			aFunctionPrimes.push_back(Vector<AFunction>());
			for (const unsigned int &id : vector)
			{
				aFunctions[aFunctions.size() - 1].push_back(availableAFunctions[id]);
				aFunctionPrimes[aFunctionPrimes.size() - 1].push_back(availableAFunctionPrimes[id]);
			}
		}
	}
};

#endif // NEURAL_NET_H