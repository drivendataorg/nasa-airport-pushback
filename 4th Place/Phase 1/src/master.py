"""
a script that generates encoders and model for prediction
"""

if __name__ == "__main__":
    import os

    # preprocessing all the data
    exec(open(os.path.join(os.path.dirname(__file__), "preprocessing.py")).read())

    # generate encoders
    exec(open(os.path.join(os.path.dirname(__file__), "encoders.py")).read())

    # train model
    exec(open(os.path.join(os.path.dirname(__file__), "train.py")).read())
