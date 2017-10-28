#include "main.h"

int main(int argc, char *argv[])
{
	std::ifstream input("trainingSet.txt");
	
	std::vector<unsigned int> numbers;
	std::vector<std::vector<double>> inputs;
	std::vector<std::vector<double>> outputs;
	for(std::string line; getline(input, line);)
	{
		inputs.push_back(std::vector<double>());
		outputs.push_back(std::vector<double>());
		unsigned int counter = 0;
		std::size_t position = 0;
		while((position = line.find(",")) != std::string::npos)
		{
			if(counter == 0)
			{
				outputs[outputs.size() - 1].push_back(std::stoi(line.substr(0, position)));
			}
			else
			{
				inputs[inputs.size() - 1].push_back(std::stoi(line.substr(0, position)) / 255.0);
			}
			++counter;
			line.erase(0, position + 1);
		}
	}
	input.close();

	NeuralNet net = NeuralNet({ inputs.size(), 64, 64, 64, outputs.size() },
	{ std::vector<unsigned int>(64, NeuralNet::Logistic),
		std::vector<unsigned int>(64, NeuralNet::Logistic),
		std::vector<unsigned int>(64, NeuralNet::Logistic),
		std::vector<unsigned int>(outputs.size(), NeuralNet::Logistic) });
	
	for(unsigned int itEpoche = 0; itEpoche < 2; ++itEpoche)
	{
		for(unsigned int itTraining = 0; itTraining < inputs.size(); ++itTraining)
		{
			net.train(inputs[itTraining], outputs[itTraining]);
		}
		std::cout << "Epoche: " << itEpoche << std::endl;
	}
	
	for (unsigned int itTraining = 0; itTraining < inputs.size(); ++itTraining)
	{
		net.forward(inputs[itTraining]);
		for (int i = 0; i < net.getOutputs().size(); ++i)
		{
			std::cout << net.getOutputs()[i] << std::endl;
		}
	}

	inputs.clear();
	input.open("testSet.txt");
	for (std::string line; getline(input, line);)
	{
		inputs.push_back(std::vector<double>());
		std::size_t position = 0;
		while ((position = line.find(",")) != std::string::npos)
		{
			inputs[inputs.size() - 1].push_back(std::stoi(line.substr(0, position)) / 255.0);
			line.erase(0, position + 1);
		}
	}
	input.close();

	for (std::vector<double> netInput : inputs)
	{
		net.forward(netInput);
		for (int i = 0; i < net.getOutputs().size(); ++i)
		{
			std::cout << net.getOutputs()[i] << std::endl;
		}
	}

	std::cin.get();
	return 0;
}