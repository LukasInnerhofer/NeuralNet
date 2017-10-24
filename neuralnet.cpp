#include "neuralnet.h"

std::vector<std::function<double(const double &)>> NeuralNet::availableActivationFunctions = std::vector<std::function<double(const double &)>> {
	[](const double &x) -> double { return 1 / (1 + std::exp(-x)); },			// Logistic
	[](const double &x) -> double { return 2 / (1 + std::exp(-2 * x)) - 1; }	// Hyperbolic Tangent (TanH)
};
std::vector<std::function<double(const double &)>> NeuralNet::availableActivationFunctionPrimes = std::vector<std::function<double(const double &)>>{
	[](const double &x) -> double { return std::exp(-x) / std::pow(1 + std::exp(-x), 2); },				// Logistic
	[](const double &x) -> double { return 4 * std::exp(-2 * x) / std::pow(1 + std::exp(-2 * x), 2); }	// Hyperbolic Tangent (TanH)
};

std::vector<double> NeuralNet::vectorFunction(std::vector<std::function<double(const double &)>> functions, const std::vector<double> &args)
{
	std::vector<double> returnValues;
	for (unsigned int it = 0; it < args.size(); ++it)
	{
		returnValues.push_back(functions[it](args[it]));
	}
	return returnValues;
}

std::vector<double> NeuralNet::getOutputs()
{
	return outputs[outputs.size() - 1];
}

NeuralNet::NeuralNet()
{
	inputs = outputs = std::vector<std::vector<double>>();
	activationFunctions = activationFunctionPrimes = std::vector<std::vector<std::function<double(const double &)>>>();
	synapses = std::vector<std::vector<std::vector<double>>>();
	
	distribution = std::uniform_real_distribution<double>(0.0, 1.0);	
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}

NeuralNet::NeuralNet(const std::vector<unsigned int> &neurons, const std::vector<std::vector<unsigned int>> &activationFunctions) : NeuralNet()
{	
	try
	{
		const std::invalid_argument badTopology("Neuron topology doesn't match activation function topology.");
		if (neurons.size() != activationFunctions.size() + 1)
		{
			throw badTopology;
		}

		for (size_t itLayers = 0; itLayers < neurons.size(); ++itLayers) 	// For each layer of neurons
		{
			std::vector<double> layerInputs;
			std::vector<std::function<double(const double &)>> layerFunctions, layerFunctionPrimes;
			std::vector<std::vector<double>> layerSynapses;	// Store all Synapses connecting this layer to the next
			for (unsigned int itNeurons = 0; itNeurons < neurons[itLayers]; ++itNeurons)	// For each Neuron in this layer
			{
				if (itLayers > 0)
				{
					layerFunctions.push_back(availableActivationFunctions[activationFunctions[itLayers - 1][itNeurons]]);
					layerFunctionPrimes.push_back(availableActivationFunctionPrimes[activationFunctions[itLayers - 1][itNeurons]]);
				}
				else
				{
					layerFunctions.push_back([](const double &x) -> double { return x; });
					layerFunctionPrimes.push_back([](const double &x) -> double { return 1; });
				}
				layerInputs.push_back(0.0);

				if (itLayers < neurons.size() - 1)	// For each layer except the output layer
				{
					std::vector<double> newSynapses;	// Store all synapses connecting this neuron to the next layer
					for (unsigned int itSynapses = 0; itSynapses < neurons[itLayers + 1]; ++itSynapses)	// For each synapse of this neuron
					{
						newSynapses.push_back(randomReal());
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
			this->activationFunctions.push_back(layerFunctions);
			activationFunctionPrimes.push_back(layerFunctionPrimes);
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
			this->inputs[itLayers] = synapses[itLayers - 1] * this->outputs[itLayers - 1];
			this->outputs[itLayers] = vectorFunction(activationFunctions[itLayers], this->inputs[itLayers]);
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
		auto errors = std::vector<std::vector<double>>(this->inputs.size() - 1, std::vector<double>());
		
		for (int itLayers = this->inputs.size() - 1; itLayers > 0; --itLayers)
		{
			if (itLayers == this->inputs.size() - 1)	// Output layer
			{
				errors[itLayers - 1] = matrixMath::hadamard(this->outputs[itLayers] - outputs, vectorFunction(activationFunctionPrimes[itLayers], this->inputs[itLayers]));
			}
			else
			{
				errors[itLayers - 1] = matrixMath::hadamard(matrixMath::transpose(synapses[itLayers]) * errors[itLayers], vectorFunction(activationFunctionPrimes[itLayers], this->inputs[itLayers]));
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
		}
	}
	catch (const std::invalid_argument &e)
	{
		std::cerr << "Invalid argument. " << e.what() << std::endl;
		return;
	}
}