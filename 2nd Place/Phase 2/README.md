# Solution - NASA Pushback to the Future Competition (Phase 2)

Caltech "Moles" team respository for predicting pushback times at US airports using federated learning.

[Competition website &rarr;](https://www.drivendata.org/competitions/218/competition-nasa-airport-pushback-phase2/page/766/)

## Overview
This repository contains code to create and execute a model to predict pushback time (defined as the time between when an airplane arives at and departs from the gate) as specified by the 2023 NASA Pushback to the Future Competition (Phase 2). Given public data from ten US airports and private data from 25 airline clients, a CatBoost model is trained for each client and the encoded predictions are fed into a 1D-CNN that is trained in a federated manner using the Flower framework to maintain the privacy of client data. We follow the results of [Ma et al.](https://arxiv.org/abs/2304.07537) to federate our gradient boosting tree-based model. 

## Repository Structure
This repository contains separate pipelines for model training and inference. Trained CNN models are stored in the "models" folder and CatBoost trees are stored in the "trees" folder.

```
nasa-pushback-competition
├── README.md
├── Pushback_Data ---------> holds raw data
├── model -----------------> holds trained CNN models
├── trees -----------------> holds trained CatBoost trees
├── requirements.txt ------> lists requirements
├── setup.py --------------> makes project pip installable
└── src
    ├── __init__.py
    ├── config.py ---------> path configuration
    ├── make_dataset.py ---> runs feature extraction
    ├── client.py ---------> flower client class
    ├── server.py ---------> flower server class
    ├── run_inference.py --> pipeline for making predictions
    ├── run_training.py ---> pipeline for training model
    └── utils.py ----------> helper functions used for inference
```

## Setup

Create and activate virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```

Install the required python packages:
```
pip install -r requirements.txt
```

If using a different directory structure, update config.py file with the desired data and model paths.

## Hardware

Machine specs and time we used to run our model

* CPU: Intel Core i5
* GPU: NVIDIA GeForce GTX 780
* Memory: 64 GB
* OS: Windows
* Train duration: 3 hours
* Inference duration: Scales depending on size of dataset


## Run training

Execute feature extraction and model training pipeline:
```
python src/run_training.py
```

## Run inference

Generate predictions for custom submission format (in file Pushback_Data/submission_format.csv):
```
python src/run_inference.py
```

## Authors

[@Brian Hu](https://github.com/BrainHu42)  
[@Nika Chuzhoy](https://github.com/nikac776)
