#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>

class NeuralNet
{
private:
	struct Neuron {
		double in = 0.0, out = 0.0;
		unsigned int activationFunction;
		Neuron(double _in, double _out, unsigned int _activationFunction)
		{
			in = _in;
			out = _out;
			activationFunction = _activationFunction;
		}
	};

	std::vector<std::vector<Neuron>> neurons;
	std::vector<std::vector<std::vector<double>>> synapses;
	
	// Random
	std::uniform_real_distribution<double> distribution;
	std::default_random_engine randomEngine;
	inline double randomReal() { return distribution(randomEngine); }
	
	static std::vector<std::function<double(const double &)>> activationFunctions;

public:
	NeuralNet();
	NeuralNet(const std::vector<unsigned int> &_neurons, const std::vector<std::vector<unsigned int>> &_activationFunctions);

	static const enum activationFunction { Logistic, TanH, ActivationFunctionMax = TanH };

	void forward(const std::vector<double> &inputs);
	void train(const std::vector<double> &inputs, const std::vector<double> &outputs);
};

#endif // NEURAL_NET_H
