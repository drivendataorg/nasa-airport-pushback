# Solution - Predict Pushback Time Challenge (by Cuong_Syr team)

Username: cuongk14

## Summary

The solution is based on constructing an XGBoost regressor model for each airport separately. We perform feature engineering based on all tables to extract  informative features. The features combined with the groundtruth label  in training data is used to learn the XGBoost model. 

# Setup

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.12. 
```
conda create --name example-submission python=3.8
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Download the data from the competition page into `data/raw`

The structure of the directory before running training or inference should be:
```
example_submission
├── data
│   ├── processed      <- submission result folder.
│   └── KJFK           <- The original, immutable data dump.
├── models             <- Trained models
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── run_training.py
│   ├── run_submission.py
│   └── features.py
├── README.md          <- The top-level README for developers using this project.
├── requirements.txt   <- The requirements file for reproducing the analysis environment
```

# Hardware

The solution was run on macOS Ventura, version 13.31 with Apple M1 Pro Processor. 
- Number of CPUs: 10
- Chip: Apple M1 Pro
- Memory: 32 GB

Both training and inference were run on CPU.
- Training time: ~ 1-2 hour for one XGBoost model per airport. Total time is around 12 hours.  
- Inference time: ~3 minutes for one  aiport. Total time is 30 minutes.

# Run training

To run training from the command line: `python src/run_training.py`

```
$ python src/run_training.py --help
Usage: run_training.py [OPTIONS]

Options:
  --airport_name                  airport name
  --lr                            learning rate, default = 2.5
  
```

By default, trained model weights will be saved to `models/v5_model_{airport_name}_{learning_rate}.json`. 


# Run prediction

To run inference from the command line: `python src/run_submission.py`

```
$ python src/run_inference.py --help
Usage: run_inference.py [OPTIONS]

Options:
  --airport_name                  airport name
  --lr                            learning rate, default = 2.5
```

By default, predictions will be saved out to `data/processed/submission_{airport_name}_{learning_rate}.csv`.

--------
