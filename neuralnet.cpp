#include "neuralnet.h"

Vector<NeuralNet::AFunction> NeuralNet::availableAFunctions = {
	[](const double &x) { return x; },								// Identity
	[](const double &x) { return 1 / (1 + std::exp(-x)); },			// Logistic
	[](const double &x) { return 2 / (1 + std::exp(-2 * x)) - 1; }	// Hyperbolic Tangent (TanH)
};
Vector<NeuralNet::AFunction> NeuralNet::availableAFunctionPrimes = {
	[](const double &x) { return 1.0; },														// Identity
	[](const double &x) { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); },				// Logistic
	[](const double &x) { return 4 * std::exp(-2 * x) / std::pow(1 + std::exp(-2 * x), 2); }	// Hyperbolic Tangent (TanH)
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

Vector<double> NeuralNet::getOutputs()
{
	return outputs[outputs.size() - 1];
}

NeuralNet::NeuralNet()
{
	inputs = outputs = Matrix<double>();
	aFunctions = aFunctionPrimes = Matrix<NeuralNet::AFunction>();
	synapses = Vector<Matrix<double>>();
}

NeuralNet::NeuralNet(const Vector<unsigned int> &neurons, const Vector<Vector<unsigned int>> &aFunctions) : NeuralNet()
{	
	try
	{
		const std::invalid_argument badTopology("Neuron topology doesn't match activation function topology.");
		if (neurons.size() != aFunctions.size() + 1)
		{
			throw badTopology;
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
					layerFunctions.push_back(availableAFunctions[aFunctions[itLayers - 1][itNeurons]]);
					layerFunctionPrimes.push_back(availableAFunctionPrimes[aFunctions[itLayers - 1][itNeurons]]);
					layerBiases.push_back(randDouble(-1, 1));
				}
				else
				{
					layerFunctions.push_back(availableAFunctions[Identity]);
					layerFunctionPrimes.push_back(availableAFunctionPrimes[Identity]);
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
				synapses.push_back(layerSynapses);
			}

			inputs.push_back(layerInputs);
			outputs.push_back(layerInputs);
			biases.push_back(layerBiases);
			this->aFunctions.push_back(layerFunctions);
			aFunctionPrimes.push_back(layerFunctionPrimes);
		}
	}
	catch (const std::invalid_argument &exception)
	{
		std::cerr << "Invalid argument exception during initialization of Neural Net: " << exception.what() << std::endl;
	}
	catch (const std::exception &exception)
	{
		std::cerr << "Unknown exception during initialization of Neural Net: " << exception.what() << std::endl;
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
			this->inputs[0][itInputs] = outputs[0][itInputs] = inputs[itInputs];
		}
		for (size_t itLayers = 1; itLayers < this->inputs.size(); ++itLayers)	// For each layer
		{
			this->inputs[itLayers] = synapses[itLayers - 1] * this->outputs[itLayers - 1];
			this->outputs[itLayers] = vectorFunction(aFunctions[itLayers], this->inputs[itLayers] + biases[itLayers]);
		}
	}
	catch (const std::invalid_argument &exception)
	{
		std::cerr << "Invalid argument exception during feedforward: " << exception.what() << std::endl;
	}
	catch (const std::exception &exception)
	{
		std::cerr << "Unknown exception during feedforward: " << exception.what() << std::endl;
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
				errors[itLayers - 1] = (this->outputs[itLayers] - outputs).hadamard(vectorFunction(aFunctionPrimes[itLayers], this->inputs[itLayers]));
			}
			else
			{
				errors[itLayers - 1] = (synapses[itLayers].transpose() * errors[itLayers]).hadamard(vectorFunction(aFunctionPrimes[itLayers], this->inputs[itLayers]));
			}
		}

		for (unsigned int itLayers = 0; itLayers < synapses.size(); ++itLayers)
		{
			for (unsigned int itNeurons = 0; itNeurons < synapses[itLayers].size(); ++itNeurons)
			{
				for (unsigned int itSynapses = 0; itSynapses < synapses[itLayers][itNeurons].size(); ++itSynapses)
				{
					synapses[itLayers][itNeurons][itSynapses] -= this->outputs[itLayers][itNeurons] * errors[itLayers][itSynapses];
				}
			}
			for (unsigned int itNeurons = 0; itNeurons < biases[itLayers + 1].size(); ++itNeurons)
			{
				biases[itLayers + 1][itNeurons] -= errors[itLayers][itNeurons];
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
		int layer = randInt(0, synapses.size() - 1);
		int neuron = randInt(0, synapses[layer].size() - 1);
		int synapse = randInt(0, synapses[layer][neuron].size() - 1);
		synapses[layer][neuron][synapse] *= randDouble(0.5, 1.5);

		layer = randInt(0, biases.size() - 1);
		neuron = randInt(0, biases[layer].size() - 1);
		biases[layer][neuron] *= randDouble(0.5, 1.5);
	}
}

NeuralNet NeuralNet::breed(const NeuralNet &partner)
{
	/*auto childNeurons = Vector<unsigned int>(inputs.size(), 0);
	auto childAFunctions = Vector<Vector<unsigned int>>(aFunctions.size() - 1, Vector<unsigned int>());
	for (decltype(inputs.size()) itLayers = 0; itLayers < inputs.size(); ++itLayers)
	{
		childNeurons[itLayers] = inputs[itLayers].size();
		if (itLayers > 0)
		{
			for (decltype(inputs[itLayers].size()) itNeurons = 0; itNeurons < inputs[itLayers].size(); ++itNeurons)
			{
				childAFunctions[itLayers - 1].push_back(0);
			}
		}
	}*/

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
				/*const auto aFunctionIndex = std::find(availableAFunctions.begin(),
					availableAFunctions.end(),
					partner.aFunctions[itLayers][itNeurons]);
				child.aFunctions[itLayers][itNeurons] = availableAFunctions[aFunctionIndex];*/
			}
		}
	}

	return child;
}