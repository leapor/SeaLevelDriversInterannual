# Code for: "Drivers of interannual to decadal sea level variability in northern Europe - Data driven approach" (under review)
## by Lea Poropat and Céline Heuzé

This repository contains the implementation and experimental code for the manuscript “Drivers of interannual to decadal sea level variability in northern Europe - Data driven approach”, currently under review.

The code is released to support transparency and reproducibility. It is not intended to be a fully documented or production-ready software package.

**Aim:** Predict monthly-mean sea level from atmospheric and hydrological variables using machine learning models (feed-forward and LSTM neural networks and linear regression with temporal dependence) and quantify their contribution to the prediction with permutation feature importance.
   
   
## Workflow Overview
The full experimental pipeline consists of the following steps:

### Data processing
- data extraction - extracts data for selected time span and locations
 - SeaLevel.py - sea level from tide gauges
 - ERA5mask.py - getting ocean-land mask for ERA5 data needed for data extraction 
 - ERA5localPoint.py - MSLP, wind (u and v component), SST from ERA5
 - ERA5localInteg.py - precipitation, evaporation, runoff from ERA5
 - ERA5globalSST.py - global mean SST from ERA5
 - Antarctica.py - Antarctic runoff
 - Greenland.py - Greenland runoff and discharge
 - NAO.py - North Atlantic Oscillation index
- data processing - preparing data for training and testing ML models
 - DataProcessing.py - preparing data
 - Correlations.py - calculating correlation between input variables

### Hyperparameter tuning (only for neural networks)
- HyperparameterTuning.py - train small ensembles of networks with all hyperparameter combinations
- SelectBestHyperparameters.py - select the best hyperparameter combination

### Model training
- Train.py

### Model evaluation
- ValidationTimeseries.py - evaluate models with validation set
- TestSetEvaluation.py - evaluate models with test se

### Feature importance experiments
- FeatureImportance.py

### Postprocessing
- CombineHyperparameters.py - summarize the hyperparameter tuning results
- CombineResults.py - combine the results from all experiments and datasets
- RecalculateMetrics.py - re-calculate evaluation metrics with a shortened test set
- FindBestModel.py - find best model type and sequence length for each station

### Plotting and analysis - create figures for the manuscript
- StudyArea.py (Fig 1)
- Schematic.py (Fig 2)
- BaselineAllBox.py (Fig 3)
- BaselineBestMap.py (Fig 4)
- FeatureImportanceMap.py (Fig 5)


## Experiments
- ANN - feed-forward neural networks for all stations
- LSTM - LSTM neural networks for all stations with sequence length of 1 or 2 previous months
- LSTM2 - LSTM neural network experiments for all stations with the other sequence length (1-2)
- LSTMtzd - LSTM experiments for stations in the Baltic Sea - North Sea transition zones with sequence length 3-12
- LSTMd - LSTM experiments for the remaining stations (outside the transition zone) with sequence length 3, 4, 5, 6, and 12 previous months
- LinReg - linear regression with temporal dependence for all stations


## Environment Setup
Code is written in Python. Two separate environments are used:
1. **ML environment** (environmentSLD-ML)
   Used for hyperparameter tuning, model training, validation, and testing. 
   Python 3.8.10 and TensorFlow 2.13.1
2. **Pre/Post-processing environment** (environmentSLD-vis)
   Used for data preprocessing, analysis, and figure generation.
