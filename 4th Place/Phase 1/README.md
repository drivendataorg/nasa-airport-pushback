# Solution - Pushback to the Future



Created by the team FLHuskies2 out of the University of Washington Tacoma.

## Setup

1. This solution with the following steps is guaranteed to run on x64 Ubuntu Server 20.04. Although it is very 
likely to run on other operating systems as well.
2. Install Python 3.10.9
3. Install the following packages manually with pip:
    
   - `pandas==1.5.3`
   - `lightgbm==3.3.5`
   - `numpy==1.24.2`
   - `pandarallel==1.6.4`
   - `tqdm==4.65.0`
   - `scikit-learn==1.2.2`

   or use `pip install -r requirements.txt`

4. Ensure that the "data" directory is located and formatted as specified in data/README.md

## Download Pretrained Models
The pretrained models and encoders can be downloaded [here](https://www.dropbox.com/scl/fo/6nparyuy3vo10j6cpho8e/h?dl=0&rlkey=eo93lv5m16q5vyyve2pqjyukk) and should be placed in the src folder.

## Run Training
Run the script `master.py`, it will likely take many hours to complete, 
but will execute the entire pipeline, from raw data to the models.

It will output 2 files, `models.pickle` and `encoders.pickle`. These will take about 500mb of storage. In the process, the train tables will also be generated and saved, these will take about 25 gb of storage.


## Run Inference
1. Obtain a variable that contains the result of the `load_model()` function in `solution.py`, this function requires 
a path to a folder that contains `models.pickle` and `encoders.pickle` as the argument, it will return a tuple that
contains the loaded models and column encoders.
2. The function to make a prediction is `predict()` in `solution.py`. It makes predictions for any number of flights,
but for only one timestamp and airport at a time. 
3. `predict()` is formatted just as in the prescreened arena submission. It requires, among other things:
   - `models`: the tuple of models and encoders obtained in step 1
   - the raw data tables filtered by timestamp between the prediction time and 30 hours prior
   - `partial_submission_format`: a dataframe of the flights and timestamps to make predictions for
4. Call predict with the required inputs and what will be returned is the `partial_submission_format` DataFrame
with the predictions in the `minutes_until_pushback` column.



