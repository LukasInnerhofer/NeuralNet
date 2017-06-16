#include "neuralnet.h"

NeuralNet::NeuralNet()
{
	neurons = std::vector<std::vector<double>>();
	synapses = std::vector<std::vector<std::vector<double>>>();
	
	distribution = std::uniform_real_distribution<double>(0.0, 1.0);	
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}

NeuralNet::NeuralNet(std::vector<unsigned int> topology) : NeuralNet()
{	
	for(int itLayers = 0; itLayers < topology.size(); ++itLayers) 	// For each layer of neurons
	{
		std::vector<double> newNeurons(topology[itLayers], 0.0);
		neurons.push_back(newNeurons);
		
		if(itLayers < topology.size() - 1) 	// For each layer of neurons except the output layer
		{
			std::vector<std::vector<double>> layerSynapses;	// Store all the synapses in this layer
			for(int itNeurons = 0; itNeurons < topology[itLayers]; ++itNeurons)	// For each neuron in this layer
			{
				std::vector<double> newSynapses;	// Store each synapse for this neuron
				for(int itSynapses = 0; itSynapses < topology[itLayers + 1]; ++itSynapses)	// For each synapse of this neuron
				{
					newSynapses.push_back(randomReal());
				}
				layerSynapses.push_back(newSynapses);
			}
			synapses.push_back(layerSynapses);
		}
	}
}

void NeuralNet::forward(std::vector<double> inputs)
{
	if(inputs.size() > neurons[0].size())
	{
		std::cerr << "Error: Number of input values exceeds number of input neurons.\n";
		return;
	}
	for(int itInputs = 0; itInputs < inputs.size(); ++itInputs)	// Load inputs into input neurons
	{
		neurons[0][itInputs] = inputs[itInputs];
	}
	for(int itLayers = 1; itLayers < neurons.size(); ++itLayers)	// For each layer
	{
		for(int itNeurons = 0; itNeurons < neurons[itLayers].size(); ++itNeurons)	// For each neuron of this layer
		{
			double sum = 0.0;	// Sum of all the incoming synapses
			for(int itPreviousNeurons = 0; itPreviousNeurons < neurons[itLayers - 1].size(); ++itPreviousNeurons)	// For each neuron of the previous layer
			{
				// Sum up the values of all the neurons of the previous layer with the weights of the synapses that connect them with this neuron
				sum += neurons[itLayers - 1][itPreviousNeurons] * synapses[itLayers - 1][itPreviousNeurons][itNeurons];	
			}
			neurons[itLayers][itNeurons] = sigmoid(sum);	// Apply the activation function
		}
	}
}