#include <iostream>
#include "MNN/Interpreter.hpp"

int main() {
    std::string path = "/Users/daniel/Downloads/gsep.mnn";

    MNN::Interpreter* interpreter = MNN::Interpreter::createFromFile(path.c_str());

    MNN::ScheduleConfig conf;
    MNN::Session* session = interpreter->createSession(conf);

    interpreter->runSession(session);

    std::cout << "Inferenced" << std::endl;
    return 0;
}