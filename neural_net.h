#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <random>

#include "matrix.h"
#include "vector.h"

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

	static std::map<std::string, AFunction> availableAFunctions;
	static std::map<std::string, AFunction> availableAFunctionPrimes;

	Vector<double> vectorFunction(Vector<AFunction> functions, const Vector<double> &args);

	int randInt(const int &lowerBound, const int &upperBound);
	double randDouble(const double &lowerBound, const double &upperBound);

public:
	inline Vector<double> NeuralNet::getOutputs() const { return this->outputs[outputs.size() - 1]; }
	inline Matrix<double> getNeurons() const { return this->outputs; }
	inline Vector<Matrix<double>> getSynapses() const { return this->synapses; }

	NeuralNet();
	NeuralNet(const Vector<unsigned int> &neurons, const Vector<Vector<std::string>> &aFunctions);

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
				const unsigned int id = std::find(availableAFunctions.begin(), availableAFunctions.end(), function) 
					- availableAFunctions.begin();
				aFunctionIds[aFunctionIds.size() - 1].push_back(id);
			}
		}

		ar(CEREAL_NVP(inputs), CEREAL_NVP(outputs), CEREAL_NVP(biases), cereal::make_nvp("activationFunctions", aFunctionIds), 
			CEREAL_NVP(synapses));
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