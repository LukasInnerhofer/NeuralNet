#include "neuralnet.h"

std::vector<std::function<double(const double &)>> NeuralNet::availableActivationFunctions = std::vector<std::function<double(const double &)>> {
	[](const double &x) -> double { return 1 / (1 + std::exp(-x)); },			// Logistic
	[](const double &x) -> double { return 2 / (1 + std::exp(-2 * x)) - 1; }	// Hyperbolic Tangent (TanH)
};
std::vector<std::function<double(const double &)>> NeuralNet::availableActivationFunctionDerivatives = std::vector<std::function<double(const double &)>>{
	[](const double &x) -> double { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); },				// Logistic
	[](const double &x) -> double { return 4 * std::exp(-2 * x) / std::pow(1 + std::exp(-2 * x), 2); }	// Hyperbolic Tangent (TanH)
};

/*std::vector<double> NeuralNet::getInputs(const unsigned int &layer)
{
	std::vector<double> inputs;
	for (Neuron neuron : neurons[layer])
	{
		inputs.push_back(neuron.in);
	}
	return inputs;
}
std::vector<double> NeuralNet::getOutputs(const unsigned int &layer)
{
	std::vector<double> outputs;
	for (Neuron neuron : neurons[layer])
	{
		outputs.push_back(neuron.out);
	}
	return outputs;
}
std::vector<std::function<double(const double &)>> NeuralNet::getActivationFunctionDerivatives(const unsigned int &layer)
{
	std::vector<std::function<double(const double &)>> derivatives;
	for (Neuron neuron : neurons[layer])
	{
		derivatives.push_back(activationFunctionDerivatives[neuron.activationFunction]);
	}
	return derivatives;
}*/
std::vector<double> NeuralNet::vectorFunction(std::vector<std::function<double(const double &)>> functions, const std::vector<double> &args)
{
	std::vector<double> returnValues;
	for (unsigned int it = 0; it < args.size(); ++it)
	{
		returnValues.push_back(functions[it](args[it]));
	}
	return returnValues;
}

NeuralNet::NeuralNet()
{
	//neurons = std::vector<std::vector<Neuron>>();
	inputs = outputs = std::vector<std::vector<double>>();
	activationFunctions = activationFunctionDerivatives = std::vector<std::vector<std::function<double(const double &)>>>();
	synapses = std::vector<std::vector<std::vector<double>>>();
	
	distribution = std::uniform_real_distribution<double>(0.0, 1.0);	
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}

NeuralNet::NeuralNet(const std::vector<unsigned int> &_neurons, const std::vector<std::vector<unsigned int>> &_activationFunctions) : NeuralNet()
{	
	try
	{
		const std::invalid_argument badTopology("Neuron topology doesn't match activation function topology.");
		if (_neurons.size() != _activationFunctions.size() + 1)
		{
			throw badTopology;
		}

		for (size_t itLayers = 0; itLayers < _neurons.size(); ++itLayers) 	// For each layer of neurons
		{
			if (_activationFunctions[itLayers].size() != _neurons[itLayers])
			{
				throw badTopology;
			}

			std::vector<double> layerInputs;
			std::vector<std::function<double(const double &)>> layerFunctions, layerFunctionDerivatives;
			std::vector<std::vector<double>> layerSynapses;	// Store all Synapses connecting this layer to the next
			for (unsigned int itNeurons = 0; itNeurons < _neurons[itLayers]; ++itNeurons)	// For each Neuron in this layer
			{
				if (_activationFunctions[itLayers][itNeurons] > ActivationFunctionMax)
				{
					throw std::invalid_argument("Activation function topology contains non-existent activation function.");
				}

				if (itLayers > 0)
				{
					layerFunctions.push_back(availableActivationFunctions[_activationFunctions[itLayers][itNeurons]]);
					layerFunctionDerivatives.push_back(availableActivationFunctionDerivatives[_activationFunctions[itLayers][itNeurons]]);
				}
				else
				{
					layerFunctions.push_back([](const double &x) -> double { return x; });
					layerFunctionDerivatives.push_back([](const double &x) -> double { return 1; });
				}
				layerInputs.push_back(0.0);

				if (itLayers < _neurons.size() - 1)	// For each layer except the output layer
				{
					std::vector<double> newSynapses;	// Store all synapses connecting this neuron to the next layer
					for (unsigned int itSynapses = 0; itSynapses < _neurons[itLayers + 1]; ++itSynapses)	// For each synapse of this neuron
					{
						newSynapses.push_back(randomReal());
					}
					layerSynapses.push_back(newSynapses);
				}
			}

			if (itLayers < _neurons.size() - 1)
			{
				synapses.push_back(layerSynapses);
			}

			inputs.push_back(layerInputs);
			outputs.push_back(layerInputs);
			activationFunctions.push_back(layerFunctions);
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

void NeuralNet::forward(const std::vector<double> &inputs)
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
			for (size_t itNeurons = 0; itNeurons < this->inputs[itLayers].size(); ++itNeurons)	// For each neuron of this layer
			{
				double *currentInput = &this->inputs[itLayers][itNeurons];
				double *currentOutput = &this->outputs[itLayers][itNeurons];
				std::function<double(const double &)> currentFunction = activationFunctions[itLayers][itNeurons];
				for (size_t itPreviousNeurons = 0; itPreviousNeurons < this->inputs[itLayers - 1].size(); ++itPreviousNeurons)	// For each neuron of the previous layer
				{
					// Sum up the values of all the neurons of the previous layer with the weights of the synapses that connect them with this neuron
					*currentInput += outputs[itLayers - 1][itPreviousNeurons] * synapses[itLayers - 1][itPreviousNeurons][itNeurons];
				}

				*currentOutput = currentFunction(*currentInput);	// Apply the activation function
			}
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

void NeuralNet::train(const std::vector<double> &inputs, const std::vector<double> &outputs)
{
	try
	{
		if (outputs.size() != this->outputs[this->outputs.size() - 1].size())
		{
			throw std::invalid_argument("Number of output values in the training set exceeds number of output neurons.");
		}

		forward(inputs);
		std::vector<std::vector<double>> errors = std::vector<std::vector<double>>(this->inputs.size(), std::vector<double>());
		
		for (int itLayers = this->inputs.size() - 1; itLayers >= 0; --itLayers)
		{
			if (itLayers == this->inputs.size() - 1)	// Output layer
			{
				errors[itLayers] = matrixMath::hadamard(this->outputs[itLayers] - outputs, vectorFunction(activationFunctionDerivatives[itLayers], this->inputs[itLayers]));
			}
		}
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "Invalid argument. " << e.what() << std::endl;
		return;
	}
}