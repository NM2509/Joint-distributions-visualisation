# Joint-distributions-visualisation

Once run, a local web-page is created showing a default version of 3D joint distributions visualisations. 
The aim of this work is to provide intuition behind joint distribution formulas that are taught in undergraduate courses on Statistics.
You can choose two of the distributions from Normal, Exponential, t-distribution and Gamma, as well as set their parameters. 
The resulting 3D visualisation adjusts automatically based on user input. 

This work is one example from the Visualisation of Mathematical Concepts series, that has been adjusted for an ease of use. 
Other examples from the series include visualisations of 3D Markov Chains, more enhanced version of joint distributions that include correlation parameter, and other concepts in optimisation and statistics.

## Dependencies

To run the code from this repository, please ensure you have the following Python libraries installed:

### Numerical Computations
* numpy 

### Scientific Computations
* scipy

### Visualisation
* plotly

### Building web application
* dash

### Bootstrap component of Dash
* dash-bootstrap-components

Note: Some libraries are imported with specific abbreviations or modules in the code, like `import numpy as np` or `from dash import dcc, html`

## How to Run

1. **Setup**: 
    - Clone the repository to your local machine
    - Ensure you've installed all required dependencies (see the "Dependencies" section)
    - Note: Some libraries are imported with specific abbreviations in the code, like `import numpy as np`

2. **Executing the Code**: 
    - Navigate to the project directory.
    - Run Python code in "Joint distributions visualisation.py" which will create a local web page
    - Choose distributions from the drop down menu and set parameters in the input boxes
    - The resulting joint distribution will appear / change automatically with user inputs
