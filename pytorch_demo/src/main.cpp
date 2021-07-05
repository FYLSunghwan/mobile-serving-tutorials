#include <iostream>
#include <chrono>
#include "torch/script.h"

using Time = std::chrono::steady_clock::time_point;

int main() {
    std::string path = "/Users/daniel/Downloads/mobile_model-4.pt";

    torch::jit::script::Module mModule;
    
    try {
        mModule = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    mModule.eval();

    float input_data = 15;

    Time st = std::chrono::steady_clock::now();
    std::vector<torch::jit::IValue> inputTensors;
    inputTensors.emplace_back(torch::zeros({1, 2, 2049}));
    inputTensors.emplace_back(torch::zeros({1, 2, 2049}));
    inputTensors.emplace_back(torch::zeros({3, 2, 1, 512}));
    auto output = mModule.forward(inputTensors).toTuple();
    Time ed = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();

    //std::cout << "Inferenced : " << output.item<float>() << std::endl;
    std::cout << "Time : " << time << " ms" << std::endl;
    return 0;
}