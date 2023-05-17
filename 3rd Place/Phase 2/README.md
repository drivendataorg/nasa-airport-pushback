# Pushback to the Future: Predict Pushback Time at US Airports

## Contributors
- Suraj Rajendran
- Matthew Love
- Prathic Sundararajan

## Summary
This solution expands on our Phase 1 solution to add federation. We train a federated ANN model and ensemble with an XGBoost model trained only on public features. The federated ANN model is trained on both public and privately generated features.

## Setup
0. Anaconda is used as an environment manager and is needed for setup. See https://docs.anaconda.com/free/anaconda/install/index.html for details.
1. Create a new conda environment environment  
    a. If using a Linux System:  
    ``` conda env create -f pushback_plane_tf_deb.yml```  
    b. If using a MaxOS System:  
    ``` conda env create -f pushback_plane_tf_mac.yml```  
2. The structure for the project directories should be as described below:  
```
.  
├── README.md                                       <- The top-level README for developers using this project.
│
├── train.py                                        <- Runs the training function to generate the models and data
│
├── predict.py                                      <- Running inference using federated and public models
├── config.py                                       <- Has configuration information for running files 
├── utils.py                                        <- Helper functions to extract features and train models 
├── Feature_Processing-TaxiTimeToGate.ipynb         <- Pre-Processes Taxi Time to Gate Feature  
│  
├── pushback_plane_tf_deb.yml                      
├── pushback_plane_tf_mac.yml  
│  
├── data  
│   │  
│   ├── submission_data.csv                         <- Labels used when Running Inference  
│   └── public_features_engineered                  <- This file structure is used per airport for data during training and inference
│   │       ├── processed_public_features_airport.pkl
│   ├──
│   └── raw                                         <- Raw data and labels for all airports
│           ├── phase2_train_labels_airport.csv.bz2 
│           │ 
│           ├── private
│           │       ├── Airport_Name  
│           │              ├── Airport_Name_config.csv  
│           │              ├── Airport_Name_etd.csv  
│           │              ├── Airport_Name_first_position.csv  
│           │              ├── Airport_Name_lamp.csv  
│           │              ├── Airport_Name_mfs.csv  
│           │              ├── Airport_Name_runways.csv  
│           │              ├── Airport_Name_standtimes.csv  
│           │              ├── Airport_Name_tbfm.csv  
│           │              └── Airport_Name_tfm.csv  
│           │
│           │
│           ├── private
│                  ├── Airport_Name  
│                          ├── Airport_Name_config.csv  
│                          ├── Airport_Name_etd.csv  
│                          ├── Airport_Name_first_position.csv  
│                          ├── Airport_Name_lamp.csv  
│                          ├── Airport_Name_mfs.csv  
│                          ├── Airport_Name_runways.csv  
│                          ├── Airport_Name_standtimes.csv  
│                          ├── Airport_Name_tbfm.csv  
│                          └── Airport_Name_tfm.csv  
│  
├── Models                                       <- ANN and XGB Models will be saved in this directory
│   └── ann_models
│   └── xgb_models                          
│  
├── predictions                                  <- Directory where predictions from predict.py will be saved
```

## General Information
To run the files, see below:
```
$ python train.py
$ python predict.py
All configuration and paths are editable in the config.py file
```

## Training
Ensure that data is in the file structrue described in the Setup section.

To train the models from command line using the automation script:     
```python train.py -t```

## Inference
Ensure that data is in the file structrue described in the Setup section.

To use the models for prediction from command line using the automation script:      
```python predict.py -i```
