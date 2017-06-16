#include "main.h"

int main(int argc, char *argv[])
{
	NeuralNet net = NeuralNet({3, 3, 3});
	net.forward({0.5, 0.5, 0.5});
	
	return 0;
}