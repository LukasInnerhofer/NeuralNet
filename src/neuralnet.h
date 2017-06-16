#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

class NeuralNet
{
private:
	std::vector<std::vector<double>> neurons;
	std::vector<std::vector<std::vector<double>>> synapses;
	
	// Random
	std::uniform_real_distribution<double> distribution;
	std::default_random_engine randomEngine;
	inline double randomReal() { return distribution(randomEngine); }
	
	static inline double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
	static inline double sigmoidPrime(double x) { return std::exp(-x) * std::pow(sigmoid(x), 2); }

public:
	NeuralNet();
	NeuralNet(std::vector<unsigned int> topology);

	void forward(std::vector<double> inputs);
	void backward();
};

#endif // NEURAL_NET_H
