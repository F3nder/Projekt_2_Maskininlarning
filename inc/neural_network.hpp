#pragma once

#include <iomanip> // For setting number of decimals for printing
#include <iostream> // For printing
#include <type_traits>
#include <vector>

#include <dense_layer.hpp>
#include <utils.hpp>

namespace yrgo {
namespace machine_learning {

    class NeuralNetwork {

    public: 
    NeuralNetwork(void) = delete;

    NeuralNetwork(const std::size_t num_inputs, 
                  const std::size_t num_hidden_nodes, 
                  const std::size_t num_outputs,
                  const ActFunc act_func_hidden = ActFunc::kRelu, 
                  const ActFunc act_func_output = ActFunc::kRelu);

    std::size_t NumInputs(void) const { return hidden_layer_.NumWeightsPerNode(); }

    std::size_t NumHiddenNodes(void) const { return hidden_layer_.NumNodes(); }

    std::size_t NumOutputs(void) const { return output_layer_.NumNodes(); }

    std::size_t NumTrainingSets(void) const {return train_order_.size(); }

    bool AddTrainingData(const std::vector<std::vector<double>>& train_input,
                         const std::vector<std::vector<double>>& train_output);
    
    bool Train(const std::size_t num_epochs, const double learning_rate = 0.01);

    const std::vector<double>& Predict(const std::vector<double> input);

    void PrintPredictions(const std::vector<std::vector<double>> input_sets,
                          const std::size_t num_decimals = 0,
                          std::ostream& ostream = std::cout);


    private:
    void CheckNumTrainingSets(void);
    void InitTrainOrderVector(void);
    void RandomizeTrainingOrder(void); 
    void Feedforward(const std::vector<double>& input);
    void Backpropagate(const std::vector<double>& reference);
    void Optimize(const std::vector<double>& input, const double learning_rate);

    DenseLayer hidden_layer_;
    DenseLayer output_layer_;
    std::vector<std::vector<double>> train_input_{};
    std::vector<std::vector<double>> train_output_{};
    std::vector<std::size_t> train_order_{}; 

    };
} // namespace machine_learning
} // namespace yrgo
