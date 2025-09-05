# Transmission Line Parameter Prediction using Neural Networks

This project provides a dual approach to analyzing electrical transmission lines. It combines a robust **analytical simulation** based on fundamental physics with a **machine learning model** trained to predict the line's electrical behavior from its physical properties.

The core of the project is a Python script that can:

1.  Calculate key parameters like Characteristic Impedance ($Z\_0$), SWR, and Input Impedance ($Z\_{in}$) for a given transmission line.
2.  Generate a synthetic dataset of transmission line configurations.
3.  Train a Neural Network (MLP Regressor) on this dataset.
4.  Use the trained model to predict line parameters and compare them against the analytical ground truth.

-----

## Key Features

  * **Analytical Engine:** A `TransmissionLine` class for accurate, physics-based calculations.
  * **Parameter Calculation:** Computes Characteristic Impedance ($Z\_0$), Propagation Constant ($\\gamma$), Attenuation ($\\alpha$), Phase Constant ($\\beta$), SWR, and Input Impedance ($Z\_{in}$).
  * **Data Generation:** A function to create large, randomized datasets for machine learning applications.
  * **ML Model:** Implements a Multi-Layer Perceptron (MLP) Regressor from `scikit-learn` to predict secondary parameters.
  * **Validation Interface:** An analysis window that directly compares the model's predictions with the ground truth results for a given set of inputs.
  * **Visualization:** Generates plots of voltage and current standing wave patterns along the line.

-----

## Requirements & Installation

The project is built using standard Python data science and numerical computing libraries.

You'll need the following libraries:

  * **NumPy:** For numerical operations.
  * **Pandas:** For data manipulation and handling the dataset.
  * **Scikit-learn:** For training the neural network and data scaling.
  * **Matplotlib:** For plotting the standing wave patterns.

You can install all the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

-----

## How to Use

The script is designed to be run from top to bottom, performing a sequence of tasks from data generation to final analysis.

### 1\. Dataset Generation

When the script is executed, the main block (`if __name__ == "__main__":`) first calls the `generate_dataset_with_distance()` function. This creates a CSV file named `transmission_line_scalar_dataset.csv` containing thousands of randomly generated transmission line examples. This file is then used for training the model.

### 2\. Model Training

The script proceeds to:

1.  Load the generated dataset.
2.  Split the data into features (R, L, G, C, etc.) and targets ($Z\_0$, SWR, etc.).
3.  Scale the numerical data using `StandardScaler` for better model performance.
4.  Define and train the `MLPRegressor` neural network on the scaled training data.

### 3\. Performance Evaluation & Analysis

After training, the script evaluates the model's accuracy using the R² score. Finally, it runs the `analysis_window()` function with a pre-defined set of `test_params`. This provides a clear, side-by-side comparison of the analytical results versus the neural network's predictions for a specific test case.

**Example Output:**
Running the analysis part will produce an output similar to this, allowing you to see how closely the ML model's predictions match reality.

```text
--- Final Design and Analysis Window ---

[1] ANALYTICAL RESULTS (Ground Truth)
 - Characteristic Impedance (Z0): 49.93 + 0.01j Ω
 - Attenuation Constant (α): 0.0081 Np/m
 - Phase Constant (β): 4.7073 rad/m
 - SWR: 1.897
 - Input Impedance (Z_in): 74.55 + 29.09j Ω

[2] NEURAL NETWORK PREDICTION
 - Characteristic Impedance (Z0): 49.89 + 0.00j Ω
 - Attenuation Constant (α): 0.00809 Np/m
 - Phase Constant (β): 4.7905 rad/m
 - SWR: 1.686
 - Input Impedance (Z_in): Not predicted by the current ML model configuration.

[3] STANDING WAVE PATTERN (Analytical)
# A plot window will appear showing the voltage and current waveforms.
```

-----

## Code Structure

  * `TransmissionLine` **class**: Contains all the physics-based formulas and methods for analytical calculations (`calculate_secondary_constants`, `calculate_swr`, `get_voltage_and_current`, etc.).
  * `generate_dataset_with_distance()` **function**: The data generation engine. It randomizes primary constants to create a diverse dataset.
  * **Model Training Block**: The section of the script where the `MLPRegressor` is defined, trained on the dataset, and its performance is initially checked.
  * `analysis_window()` **function**: The final validation tool. It takes a single set of parameters and uses both the `TransmissionLine` class and the trained model to compare their outputs.
  * **Main Execution Block**: The `if __name__ == "__main__":` block orchestrates the entire process from data generation to the final analysis.
