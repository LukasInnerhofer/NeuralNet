#include "neural_net.h"

std::map<std::string, NeuralNet::AFunction> NeuralNet::availableAFunctions = {
	{"identity", [](const double &x) { return x; }},								// Identity
	{"logistic", [](const double &x) { return 1 / (1 + std::exp(-x)); }},			// Logistic
	{"tanh", [](const double &x) { return 2 / (1 + std::exp(-2 * x)) - 1; }}		// Hyperbolic Tangent (TanH)
};
std::map<std::string, NeuralNet::AFunction> NeuralNet::availableAFunctionPrimes = {
	{"identity", [](const double &x) { return 1.0; }},														// Identity
	{"logistic", [](const double &x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); }},				// Logistic
	{"tanh", [](const double &x) { return 4 * std::exp(-2 * x) / std::pow(1 + std::exp(-2 * x), 2); }}		// Hyperbolic Tangent (TanH)
};

int NeuralNet::randInt(const int &lowerBound, const int &upperBound)
{
	std::default_random_engine randomEngine;
	std::random_device randomDevice;
	randomEngine = std::default_random_engine(randomDevice());
	auto intDistribution = std::uniform_int_distribution<int>(lowerBound, upperBound);
	return intDistribution(randomEngine);
}

double NeuralNet::randDouble(const double &lowerBound, const double &upperBound)
{
	std::default_random_engine randomEngine;
	std::random_device randomDevice;
	randomEngine = std::default_random_engine(randomDevice());
	auto doubleDistribution = std::uniform_real_distribution<double>(lowerBound, upperBound);
	return doubleDistribution(randomEngine);
}

Vector<double> NeuralNet::vectorFunction(Vector<NeuralNet::AFunction> functions, const Vector<double> &args)
{
	Vector<double> returnValues = Vector<double>();
	for (unsigned int it = 0; it < args.size(); ++it)
	{
		returnValues.push_back(functions[it](args[it]));
	}
	return returnValues;
}

NeuralNet::NeuralNet()
{
	this->inputs = this->outputs = Matrix<double>();
	this->aFunctions = this->aFunctionPrimes = Matrix<NeuralNet::AFunction>();
	this->synapses = Vector<Matrix<double>>();
}

NeuralNet::NeuralNet(const Vector<unsigned int> &neurons, const Vector<Vector<std::string>> &aFunctions) : NeuralNet()
{	
	try
	{
		if (neurons.size() != aFunctions.size() + 1)
		{
			throw std::invalid_argument("Neuron topology doesn't match activation function topology.");
		}

		for (decltype(neurons.size()) itLayers = 0; itLayers < neurons.size(); ++itLayers) 	// For each layer of neurons
		{
			Vector<double> layerInputs;
			Vector<double> layerBiases;
			Vector<NeuralNet::AFunction> layerFunctions, layerFunctionPrimes;
			Matrix<double> layerSynapses;	// Store all Synapses connecting this layer to the next
			for (unsigned int itNeurons = 0; itNeurons < neurons[itLayers]; ++itNeurons)	// For each Neuron in this layer
			{
				if (itLayers > 0)
				{
					layerFunctions.push_back(this->availableAFunctions[aFunctions[itLayers - 1][itNeurons]]);
					layerFunctionPrimes.push_back(this->availableAFunctionPrimes[aFunctions[itLayers - 1][itNeurons]]);
					layerBiases.push_back(randDouble(-1, 1));
				}
				else
				{
					layerFunctions.push_back(this->availableAFunctions["identity"]);
					layerFunctionPrimes.push_back(this->availableAFunctionPrimes["identity"]);
					layerBiases.push_back(0);
				}
				layerInputs.push_back(0.0);

				if (itLayers < neurons.size() - 1)	// For each layer except the output layer
				{
					Vector<double> newSynapses;	// Store all synapses connecting this neuron to the next layer
					for (unsigned int itSynapses = 0; itSynapses < neurons[itLayers + 1]; ++itSynapses)	// For each synapse of this neuron
					{
						newSynapses.push_back(randDouble(-1, 1));
					}
					layerSynapses.push_back(newSynapses);
				}
			}

			if (itLayers < neurons.size() - 1)
			{
				this->synapses.push_back(layerSynapses);
			}

			this->inputs.push_back(layerInputs);
			this->outputs.push_back(layerInputs);
			this->biases.push_back(layerBiases);
			this->aFunctions.push_back(layerFunctions);
			this->aFunctionPrimes.push_back(layerFunctionPrimes);
		}
	}
	catch (const std::invalid_argument &exception)
	{
		std::cerr << "Invalid argument exception during initialization of Neural Net: " << exception.what() << std::endl;
	}
}

void NeuralNet::forward(const Vector<double> &inputs)
{
	try
	{
		if (inputs.size() > this->inputs[0].size())
		{
			throw std::invalid_argument("Number of input values exceeds number of input neurons.");
		}

		for (size_t itInputs = 0; itInputs < inputs.size(); ++itInputs)	// Load inputs into input neurons
		{
			this->inputs[0][itInputs] = this->outputs[0][itInputs] = inputs[itInputs];
		}
		for (size_t itLayers = 1; itLayers < this->inputs.size(); ++itLayers)	// For each layer
		{
			this->inputs[itLayers] = this->synapses[itLayers - 1] * this->outputs[itLayers - 1];
			this->outputs[itLayers] = vectorFunction(this->aFunctions[itLayers], this->inputs[itLayers] + this->biases[itLayers]);
		}
	}
	catch (const std::invalid_argument &exception)
	{
		std::cerr << "Invalid argument exception during feedforward: " << exception.what() << std::endl;
	}
}

void NeuralNet::train(const Vector<double> &inputs, const Vector<double> &outputs)
{
	try
	{
		if (outputs.size() != this->outputs[this->outputs.size() - 1].size())
		{
			throw std::invalid_argument("Number of output values in the training set exceeds number of output neurons.");
		}

		forward(inputs);
		auto errors = Vector<Vector<double>>(this->inputs.size() - 1, Vector<double>());
		
		for (int itLayers = this->inputs.size() - 1; itLayers > 0; --itLayers)
		{
			if (itLayers == this->inputs.size() - 1)	// Output layer
			{
				errors[itLayers - 1] = 
					(this->outputs[itLayers] - outputs)
					.hadamard
					(vectorFunction(this->aFunctionPrimes[itLayers], this->inputs[itLayers]));
			}
			else
			{
				errors[itLayers - 1] = 
					(this->synapses[itLayers].transpose() * errors[itLayers])
					.hadamard
					(vectorFunction(this->aFunctionPrimes[itLayers], this->inputs[itLayers]));
			}
		}

		for (unsigned int itLayers = 0; itLayers < this->synapses.size(); ++itLayers)
		{
			for (unsigned int itNeurons = 0; itNeurons < this->synapses[itLayers].size(); ++itNeurons)
			{
				for (unsigned int itSynapses = 0; itSynapses < this->synapses[itLayers][itNeurons].size(); ++itSynapses)
				{
					this->synapses[itLayers][itNeurons][itSynapses] -= this->outputs[itLayers][itNeurons] * errors[itLayers][itSynapses];
				}
			}
			for (unsigned int itNeurons = 0; itNeurons < this->biases[itLayers + 1].size(); ++itNeurons)
			{
				this->biases[itLayers + 1][itNeurons] -= errors[itLayers][itNeurons];
			}
		}
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "Invalid argument. " << e.what() << std::endl;
		return;
	}
}

void NeuralNet::mutate()
{
	for (int it = 0; it < 64; ++it)
	{
		int layer = randInt(0, this->synapses.size() - 1);
		int neuron = randInt(0, this->synapses[layer].size() - 1);
		int synapse = randInt(0, this->synapses[layer][neuron].size() - 1);
		this->synapses[layer][neuron][synapse] *= randDouble(0.5, 1.5);

		layer = randInt(0, this->biases.size() - 1);
		neuron = randInt(0, this->biases[layer].size() - 1);
		this->biases[layer][neuron] *= randDouble(0.5, 1.5);
	}
}

NeuralNet NeuralNet::breed(const NeuralNet &partner)
{
	auto child = *this;
	
	for (decltype(child.inputs.size()) itLayers = 0; itLayers < child.inputs.size(); ++itLayers)
	{
		for (decltype(child.inputs[itLayers].size()) itNeurons = 0; itNeurons < child.inputs[itLayers].size(); ++itNeurons)
		{
			if (itLayers < child.inputs.size() - 1)
			{
				for (decltype(child.synapses[itLayers][itNeurons].size()) itOtherNeurons = 0; itOtherNeurons < child.synapses[itLayers][itNeurons].size(); ++itOtherNeurons)
				{
					if (randDouble(0, 1) < 0.5)
						child.synapses[itLayers][itNeurons][itOtherNeurons] = partner.synapses[itLayers][itNeurons][itOtherNeurons];
				}
			}
			
			if (randDouble(0, 1) < 0.5)
			{
				child.biases[itLayers][itNeurons] = partner.biases[itLayers][itNeurons];
				child.aFunctions[itLayers][itNeurons] = partner.aFunctions[itLayers][itNeurons];
			}
		}
	}

	return child;
}