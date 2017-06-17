#include "neuralnet.h"

std::vector<std::function<double(const double &)>> NeuralNet::activationFunctions = std::vector<std::function<double(const double &)>> {
	[](const double &x) -> double { return 1 / (1 + std::exp(-x)); },			// Logistic
	[](const double &x) -> double { return 2 / (1 + std::exp(-2 * x)) - 1; }	// Hyperbolic Tangent (TanH)
};

NeuralNet::NeuralNet()
{
	neurons = std::vector<std::vector<Neuron>>();
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

			std::vector<Neuron> layerNeurons;	// Store all neurons of this layer
			std::vector<std::vector<double>> layerSynapses;	// Store all Synapses connecting this layer to the next
			for (unsigned int itNeurons = 0; itNeurons < _neurons[itLayers]; ++itNeurons)	// For each Neuron in this layer
			{
				if (_activationFunctions[itLayers][itNeurons] > ActivationFunctionMax)
				{
					throw std::invalid_argument("Activation function topology contains non-existent activation function.");
				}

				if (itLayers > 0)
					layerNeurons.push_back(Neuron(0.0, 0.0, _activationFunctions[itLayers][itNeurons]));
				else
					layerNeurons.push_back(Neuron(0.0, 0.0, 0));

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

			neurons.push_back(layerNeurons);
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
		if (neurons.size() > 0)
		{
			if (inputs.size() > neurons[0].size())
			{
				std::cerr << "Error during feedforward: Number of input values exceeds number of input neurons.\n";
				return;
			}

			for (size_t itInputs = 0; itInputs < inputs.size(); ++itInputs)	// Load inputs into input neurons
			{
				neurons[0][itInputs].in = neurons[0][itInputs].out = inputs[itInputs];
			}
			for (size_t itLayers = 1; itLayers < neurons.size(); ++itLayers)	// For each layer
			{
				for (size_t itNeurons = 0; itNeurons < neurons[itLayers].size(); ++itNeurons)	// For each neuron of this layer
				{
					for (size_t itPreviousNeurons = 0; itPreviousNeurons < neurons[itLayers - 1].size(); ++itPreviousNeurons)	// For each neuron of the previous layer
					{
						// Sum up the values of all the neurons of the previous layer with the weights of the synapses that connect them with this neuron
						neurons[itLayers][itNeurons].in += neurons[itLayers - 1][itPreviousNeurons].out * synapses[itLayers - 1][itPreviousNeurons][itNeurons];
					}

					neurons[itLayers][itNeurons].out = activationFunctions[neurons[itLayers][itNeurons].activationFunction](neurons[itLayers][itNeurons].in);	// Apply the activation function
				}
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