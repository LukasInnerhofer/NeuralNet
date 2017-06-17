#include "main.h"

int main(int argc, char *argv[])
{
	NeuralNet net = NeuralNet({ 3, 3, 3 }, 
			{ std::vector<unsigned int>(3, NeuralNet::TanH), 
			std::vector<unsigned int>(3, NeuralNet::Logistic) });
	net.forward({0.5, -0.5, 0.5});
	std::cin.get();
	return 0;
}