#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>

// Function to remove the 'h' prefix from the string
std::string removeHPrefix(const std::string& str) {
    if (str.size() >= 1 && str[0] == 'h') {
        return str.substr(1);
    }
    return str;
}

// Function to convert an octal string to a uint32_t
uint32_t octalStringToUint32(const std::string& octalStr) {
    std::string trimmedStr = removeHPrefix(octalStr);

    // std::cout << trimmedStr<<" - ";

    std::stringstream ss;
    uint32_t value;
    ss << std::hex << trimmedStr;
    ss >> value;
    // std::cout << value<<" * ";
    return value;
}

// Function to read the file and parse the octal strings into uint32_t
std::vector<uint32_t> parseOctalFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    std::string line;
    std::getline(file, line);

    std::istringstream iss(line);
    std::vector<uint32_t> values;

    std::string token;
    while (iss >> token) {
        values.push_back(octalStringToUint32(token));
    }

    return values;
}

int main() {
    try {
        std::string filePath = "./testData_888/RA.txt";
        std::vector<uint32_t> values = parseOctalFile(filePath);

        for (const auto& value : values) {
            std::cout << std::hex << value << " ";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}