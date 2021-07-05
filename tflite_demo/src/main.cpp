#include <iostream>
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

int main() {
    std::string path = "/Users/daniel/Downloads/simple_model.tflite";

    auto model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    auto resolver = tflite::ops::builtin::BuiltinOpResolver();
    std::unique_ptr<tflite::Interpreter> interpreter;

    tflite::InterpreterBuilder(*model, resolver)(&interpreter, 4);
    if (!interpreter) {
        std::cerr << "Failed to initialize the interpreter";
        return -1;
    }
    interpreter->AllocateTensors();

    auto input = interpreter->typed_input_tensor<float>(0);
    auto output = interpreter->typed_output_tensor<float>(0);

    *input = 15;
    interpreter->Invoke();
    std::cout << "Inferenced : " << *output << std::endl;
    return 0;
}