#include "main.h"

int main(int argc, char *argv[])
{
	NeuralNet net = NeuralNet({ 2, 4, 4, 1 }, 
			{ std::vector<unsigned int>(4, NeuralNet::Logistic),
			std::vector<unsigned int>(4, NeuralNet::Logistic),
			std::vector<unsigned int>(1, NeuralNet::Logistic) });

	for (int i = 0; i < 4096; ++i)
	{
		net.train({ 0,0 }, { 0 });
		net.train({ 1,0 }, { 1 });
		net.train({ 0,1 }, { 1 });
		net.train({ 1,1 }, { 0 });
	}

	net.forward({ 0,0 });
	std::cout << net.getOutputs()[0] << std::endl;
	net.forward({ 1,0 });
	std::cout << net.getOutputs()[0] << std::endl;
	net.forward({ 0,1 });
	std::cout << net.getOutputs()[0] << std::endl;
	net.forward({ 1,1 });
	std::cout << net.getOutputs()[0] << std::endl;

	std::cin.get();
	return 0;
}