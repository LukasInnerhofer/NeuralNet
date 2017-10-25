#include "main.h"

int main(int argc, char *argv[])
{
	NeuralNet net = NeuralNet({ 255, 32, 32, 9 }, 
			{ std::vector<unsigned int>(32, NeuralNet::Logistic),
			std::vector<unsigned int>(32, NeuralNet::Logistic),
			std::vector<unsigned int>(9, NeuralNet::Logistic) });

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
				numbers.push_back(std::stoi(line.substr(0, position)));
				for(unsigned int itOutputs = 0; itOutputs < 9; ++itOutputs)
				{
					outputs[outputs.size() - 1].push_back(itOutputs == numbers[numbers.size() - 1] - 1);
				}
			}
			else
			{
				inputs[inputs.size() - 1].push_back(std::stoi(line.substr(0, position)) / 255.0);
			}
			++counter;
			line.erase(0, position + 1);
		}
	}
	
	for(unsigned int itEpoche = 0; itEpoche < 256; ++itEpoche)
	{
		double error = 0;
		for(unsigned int itTraining = 0; itTraining < inputs.size(); ++itTraining)
		{
			net.train(inputs[itTraining], outputs[itTraining]);
			
			for(unsigned int itOutputs = 0; itOutputs < outputs.size(); ++itOutputs)
			{
				error += outputs[itTraining][itOutputs] - net.getOutputs()[itOutputs];
			}
		}
		std::cout << "Epoche: " << itEpoche << ", Error: " << error / outputs.size() << std::endl;
	}
	
	net.forward(inputs[1]);
	for(int i = 0; i < 9; ++i)
	{
		std::cout << net.getOutputs()[i] << std::endl;
	}

	std::cin.get();
	return 0;
}