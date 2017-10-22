#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>

#include "matrixMath.h"

class NeuralNet
{
private:
	/*struct Neuron {
		double in = 0.0, out = 0.0;
		unsigned int activationFunction;
		Neuron(double _in, double _out, unsigned int _activationFunction)
		{
			in = _in;
			out = _out;
			activationFunction = _activationFunction;
		}
	};*/

	//std::vector<std::vector<Neuron>> neurons;
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> outputs;
	std::vector<std::vector<std::function<double(const double &)>>> activationFunctions;
	std::vector<std::vector<std::function<double(const double &)>>> activationFunctionDerivatives;
	std::vector<std::vector<std::vector<double>>> synapses;
	
	// Random
	std::uniform_real_distribution<double> distribution;
	std::default_random_engine randomEngine;
	inline double randomReal() { return distribution(randomEngine); }
	
	static std::vector<std::function<double(const double &)>> availableActivationFunctions;
	static std::vector<std::function<double(const double &)>> availableActivationFunctionDerivatives;

	/*std::vector<double> getInputs(const unsigned int &layer);
	std::vector<double> getOutputs(const unsigned int &layer);
	std::vector<std::function<double(const double &)>> getActivationFunctionDerivatives(const unsigned int &layer);*/
	std::vector<double> vectorFunction(std::vector<std::function<double(const double &)>> functions, const std::vector<double> &args);

public:
	NeuralNet();
	NeuralNet(const std::vector<unsigned int> &_neurons, const std::vector<std::vector<unsigned int>> &_activationFunctions);

	static const enum activationFunction { Logistic, TanH, ActivationFunctionMax = TanH };

	void forward(const std::vector<double> &inputs);
	void train(const std::vector<double> &inputs, const std::vector<double> &outputs);
};

#endif // NEURAL_NET_H