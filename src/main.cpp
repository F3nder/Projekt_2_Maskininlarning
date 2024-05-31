/********************************************************************************
 * @brief Implementation of dense layers in C++. Later on, this dense layer
 *        implementation will be used to create conventional neural networks.
 ********************************************************************************/
#include <button.hpp>
#include <led.hpp>
#include <neural_network.hpp>
#include <vector>
#include <algorithm>

using namespace yrgo::machine_learning;
using namespace yrgo::rpi;

/********************************************************************************
 * @brief Creates a neural network trained to predict a 2-bit XOR pattern.
 *        The network consists of two inputs, two hidden nodes and one output.
 *        TanH is used as activation function in the hidden layer in order 
 *        to make the neural network better at prediction complex patterns.
 *        The model is trained during 1000 epochs with a 10% learning rate.
 *        If the training is successful, the training inputs are used for'
 *        prediction, which is printed in the terminal.
 ********************************************************************************/
int main(void) {
    const std::vector<std::vector<double>> train_inputs{
        {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
        {0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
        
        {1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
        {1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}

    };
    const std::vector<std::vector<double>> train_outputs{
        {0}, {1}, {1}, {0},
        {1}, {0}, {0}, {1},

        {1}, {0}, {0}, {1},
        {0}, {1}, {1}, {0}
    };

    NeuralNetwork network(4, 10, 1, ActFunc::kTanh);
    network.AddTrainingData(train_inputs, train_outputs);
    if (network.Train(10000, 0.01)) {
        network.PrintPredictions(train_inputs);
    }

    Led Led(17);    
    Button Button1(22);
    Button Button2(23);
    Button Button3(24);
    Button Button4(25);

    std::vector<double> input{0,0,0,0};

    while(1)
    {
        input[0] = Button1.isPressed() ? 1:0;
        input[1] = Button2.isPressed() ? 1:0;
        input[2] = Button3.isPressed() ? 1:0;
        input[3] = Button4.isPressed() ? 1:0;

        if(std::count(input.begin(), input.end(), 1) %2 !=0)
        {
            Led.on();
        }
        else 
        {
            Led.off();
        }
    }

   
    return 0;
}