#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Define activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Define derivative of activation function
double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

class NeuralNetwork {
public:
    NeuralNetwork(const vector<int>& sizes) {
        // Initialize weights and biases randomly
        for (int i = 1; i < sizes.size(); i++) {
            int rows = sizes[i];
            int cols = sizes[i-1];
            weights_.push_back(vector<double>(rows * cols));
            for (int j = 0; j < rows * cols; j++) {
                weights_[i-1][j] = ((double) rand() / RAND_MAX) * 2 - 1;
            }
            biases_.push_back(vector<double>(rows));
            for (int j = 0; j < rows; j++) {
                biases_[i-1][j] = ((double) rand() / RAND_MAX) * 2 - 1;
            }
        }
    }
    
    vector<double> feedforward(const vector<double>& inputs) {
        vector<double> activations = inputs;
        for (int i = 0; i < weights_.size(); i++) {
            vector<double> weighted_sums(weights_[i].size() / biases_[i].size());
            for (int j = 0; j < weights_[i].size(); j++) {
                weighted_sums[j / biases_[i].size()] += weights_[i][j] * activations[j % biases_[i].size()];
            }
            activations.clear();
            for (int j = 0; j < biases_[i].size(); j++) {
                activations.push_back(sigmoid(weighted_sums[j] + biases_[i][j]));
            }
        }
        return activations;
    }
    
    void train(const vector<pair<vector<double>, double>>& training_data, int epochs, double learning_rate) {
        for (int i = 0; i < epochs; i++) {
            double cost = 0.0;
            for (const auto& example : training_data) {
                const vector<double>& inputs = example.first;
                double target_output = example.second;
                vector<double> activations = feedforward(inputs);
                double output = activations.back();
                double error = output - target_output;
                cost += error * error;
                vector<double> deltas = { error * sigmoid_derivative(output) };
                for (int j = weights_.size() - 1; j >= 0; j--) {
                    vector<double> next_deltas(weights_[j].size() / biases_[j].size());
                    for (int k = 0; k < biases_[j].size(); k++) {
                        double weighted_delta = 0.0;
                        for (int l = 0; l < biases_[j+1].size(); l++) {
                            weighted_delta += weights_[j][k + l * biases_[j].size()] * deltas[l];
                        }
                        next_deltas[k] = weighted_delta * sigmoid_derivative(activations[k + biases_[j].size()]);
                    }
                    deltas = next_deltas;
                    vector<double> new_biases(biases_[j].size());
                    vector<double> new_weights(weights_[j].size());
                    for (int k = 0; k < biases_[j
