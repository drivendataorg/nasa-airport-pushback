# Pushback to the Future: Predict Pushback Time at US Airports

## Contributors
- Suraj Rajendran
- Matthew Love
- Prathic Sundararajan

## Summary
This solution uses a seperate XGboost model per airport. Each Airports model consists of a regression and classification model. The regression model is used for an initial prediction of minutes_until_pushback, the classification model is then used to modify the regression output by either subtracting or adding the median of all over/underestimated values from the initial minutes_until_pushback prediction.

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
├── README.md                                    <- The top-level README for developers using this project.
│
├── run_model.py                                 <- Automates pipeline to minimize steps during training and inference
│
├── Feature_Processing-AirplaneCodes.ipynb       <- Pre-Processes Airplane Codes Feature  
├── Feature_Processing-ETD_no_PQDM.ipynb         <- Pre-Processes ETD Feature with out optimizing using PQDM  
├── Feature_Processing-ETD_PQDM.ipynb            <- Pre-Processes ETD Feature optimized using PQDM  
├── Feature_Processing-TaxiTimeToGate.ipynb      <- Pre-Processes Taxi Time to Gate Feature  
├── Run_Inference.ipynb                          <- Runs inference with a pre-trained model  
├── Train_Models.ipynb                           <- Trains Models for all Airports  
├── helper.py                                    <- Has various function defintions used throughout project   
│  
├── pushback_plane_tf_deb.yml                      
├── pushback_plane_tf_mac.yml  
│  
├── Data  
│   │  
│   ├── submission_data.csv                      <- Labels used when Running Inference  
│   └── Airport_Name                             <- This file structure is used per airport for data during training and inference
│       ├── Airport_Name  
│       │   ├── Airport_Name_config.csv  
│       │   ├── Airport_Name_etd.csv  
│       │   ├── Airport_Name_first_position.csv  
│       │   ├── Airport_Name_lamp.csv  
│       │   ├── Airport_Name_mfs.csv  
│       │   ├── Airport_Name_runways.csv  
│       │   ├── Airport_Name_standtimes.csv  
│       │   ├── Airport_Name_tbfm.csv  
│       │   └── Airport_Name_tfm.csv  
│       └── train_labels_Airport_Name.csv        <- Labels used when Training  
│  
├── Inference_Extracted_Features                 <- Pre-Processed features saved here when running inference  
│   └── Current_Features                         <- Pre-Processed features to use during inference placed here  
│  
├── Inference_Predictions                        <- Model Predictions Saved here when running inference  
│  
├── Training_Extracted_Features                  <- Pre-Processed features saved here when training  
│   └── Current_Features                         <- Pre-Processed features to use during training placed here  
│  
└── Models                                       <- Models are saved here during training  
    └── chosen                                   <- Model to run during inference placed here  
```

## General Information
The project structure and notebooks are generally setup for fast experimentation ie. exploring features and model architectures. However, in the interest of ease of use a script has been written to automate the workflow. Both methods of interacting with the project are described below. The options for the script are shown below:

```
$ python run_model.py -h
Options:
     -h: print options to console
     -d: if present will opt NOT to use PQDM to accelerate pre-processing of ETD feature
     -t: Run Training Pipeline
     -i: Run Inference Pipeline
**NOTE** if both -t and -i options are present the last argument in the command will be used.
```

## Training

Ensure that data is in the file structrue described in the Setup section.

To train the model from command line using the automation script:

```
python run_model.py -t
```

To use the manual workflow:

0. In all of the pre-processing notebook ensure that `bool_submission_prep = 0`
1. Run all cells in the Feature_Processing-AirplaneCodes.ipynb notebook
2. Choose appropriate ETD notebook and run all cells.
3. Run all cells in the Feature_Processing-TaxiTimeToGate.ipynb
4. All the data will be populated in the 'Training_Extracted_Features' directory.
5. Move the latest data for each feature in Training_Extracted_Features into the Training_Extracted_Features/Current_Features directory. The end result should look like below:

```  
Training_Extracted_Features/  
│  
└── Current_Features  
    ├── gufi_KATL_airlinecode.csv  
    ├── gufi_KCLT_airlinecode.csv  
    ├── gufi_KDEN_airlinecode.csv  
    ├── gufi_KDFW_airlinecode.csv  
    ├── gufi_KJFK_airlinecode.csv  
    ├── gufi_KMEM_airlinecode.csv  
    ├── gufi_KMIA_airlinecode.csv  
    ├── gufi_KORD_airlinecode.csv  
    ├── gufi_KPHX_airlinecode.csv  
    ├── gufi_KSEA_airlinecode.csv  
    ├── timepointgufi_KATL_etd.csv  
    ├── timepointgufi_KCLT_etd.csv  
    ├── timepointgufi_KDEN_etd.csv  
    ├── timepointgufi_KDFW_etd.csv  
    ├── timepointgufi_KJFK_etd.csv  
    ├── timepointgufi_KMEM_etd.csv  
    ├── timepointgufi_KMIA_etd.csv  
    ├── timepointgufi_KORD_etd.csv  
    ├── timepointgufi_KPHX_etd.csv  
    ├── timepointgufi_KSEA_etd.csv  
    ├── timepoint_KATL_taxitime_to_gate.csv  
    ├── timepoint_KCLT_taxitime_to_gate.csv  
    ├── timepoint_KDEN_taxitime_to_gate.csv  
    ├── timepoint_KDFW_taxitime_to_gate.csv  
    ├── timepoint_KJFK_taxitime_to_gate.csv  
    ├── timepoint_KMEM_taxitime_to_gate.csv  
    ├── timepoint_KMIA_taxitime_to_gate.csv  
    ├── timepoint_KORD_taxitime_to_gate.csv  
    ├── timepoint_KPHX_taxitime_to_gate.csv  
    └── timepoint_KSEA_taxitime_to_gate.csv  
```  

6. Run all cells in the Train_Models.ipynb. Your trained model will now be in the Models/ directory under the directory with the largest number.

## Inference

Ensure that data is in the file structrue described in the Setup section.

To train the model from command line using the automation script:

```
python run_model.py -i
```

To use the manual workflow:

0. In all of the pre-processing notebook ensure that 'bool_submission_prep = 1'
1. Run all cells in the Feature_Processing-AirplaneCodes.ipynb notebook
2. Choose appropriate ETD notebook and run all cells.
3. Run all cells in the Feature_Processing-TaxiTimeToGate.ipynb
4. All the data will be populated in the 'Training_Extracted_Features' directory.
5. Move the latest data for each feature in Training_Extracted_Features into the Inference_Extracted_Features/Current_Features directory. The end result should look like below:
    ```  
    Inference_Extracted_Features/  
    │  
    └── Current_Features  
        ├── gufi_KATL_airlinecode.csv  
        ├── gufi_KCLT_airlinecode.csv  
        ├── gufi_KDEN_airlinecode.csv  
        ├── gufi_KDFW_airlinecode.csv  
        ├── gufi_KJFK_airlinecode.csv  
        ├── gufi_KMEM_airlinecode.csv  
        ├── gufi_KMIA_airlinecode.csv  
        ├── gufi_KORD_airlinecode.csv  
        ├── gufi_KPHX_airlinecode.csv  
        ├── gufi_KSEA_airlinecode.csv  
        ├── timepointgufi_KATL_etd.csv  
        ├── timepointgufi_KCLT_etd.csv  
        ├── timepointgufi_KDEN_etd.csv  
        ├── timepointgufi_KDFW_etd.csv  
        ├── timepointgufi_KJFK_etd.csv  
        ├── timepointgufi_KMEM_etd.csv  
        ├── timepointgufi_KMIA_etd.csv  
        ├── timepointgufi_KORD_etd.csv  
        ├── timepointgufi_KPHX_etd.csv  
        ├── timepointgufi_KSEA_etd.csv  
        ├── timepoint_KATL_taxitime_to_gate.csv  
        ├── timepoint_KCLT_taxitime_to_gate.csv  
        ├── timepoint_KDEN_taxitime_to_gate.csv  
        ├── timepoint_KDFW_taxitime_to_gate.csv  
        ├── timepoint_KJFK_taxitime_to_gate.csv  
        ├── timepoint_KMEM_taxitime_to_gate.csv  
        ├── timepoint_KMIA_taxitime_to_gate.csv  
        ├── timepoint_KORD_taxitime_to_gate.csv  
        ├── timepoint_KPHX_taxitime_to_gate.csv  
        └── timepoint_KSEA_taxitime_to_gate.csv  
    ```  
6. Move all files associated with the model that are to be used to the Models/chosen/ directory.
7. Run all cells in the Run_Inference.ipynb notebook.
8. The model predictions will be under the largest numbered directory in Inference_Predictions/.
