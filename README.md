# Ventus-OpenCL-Testcase

This repository contains various test cases for the Ventus OpenCL project.

## Table of Contents

- [Introduction](#introduction)
- [TensorCoreCase](#tensorcorecase)
- [OpenCL Test Cases](#opencl-test-cases)
- [Installation and Usage](#installation-and-usage)

## Introduction

The `Ventus-OpenCL-Testcase` repository provides a collection of test cases for the Ventus OpenCL project. These test cases are designed to help developers and researchers validate and optimize their OpenCL implementations.

## TensorCoreCase

The `TensorCoreCase` directory contains Python scripts that generate data required for hardware tensor cores. The generated data is formatted to match the requirements of the registers.

### Directory Structure

- **Path**: `Ventus-OpenCL-Testcase/TensorCoreCase`
- **Description**: Python scripts for generating tensor core data.

## OpenCL Test Cases

The `_get_case` directory contains multiple OpenCL test cases. These test cases can be compiled and run after installing the LLVM-project.

### Directory Structure

- **Path**: `Ventus-OpenCL-Testcase/_get_case`
- **Description**: Multiple OpenCL test cases.

### Example Usage

1. **Navigate to the Test Case Directory**
   ```sh
   cd _get_case/mma
   ```

2. **Compile the Test Case**
   ```sh
   make
   ```

3. **Run the Test Case**
   ```sh
   ./mma.out
   ```

## Installation and Usage

### Prerequisites

Before running the test cases, ensure you have the following dependencies installed:

- **LLVM-project**: Clone and install the LLVM-project from the following repository:
  ```sh
  git clone https://github.com/THU-DSP-LAB/llvm-project.git
  cd llvm-project
  # Follow the installation instructions provided in the repository
  ```

### Running the Test Cases

1. **Navigate to the Test Case Directory**
   ```sh
   cd _get_case/<test-case-directory>
   ```

2. **Compile the Test Case**
   ```sh
   make
   ```

3. **Run the Test Case**
   ```sh
   ./<test-case>.out
   ```

---

If you have any questions or need further assistance, feel free to contact us!
