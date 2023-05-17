import getopt
import os
import shutil
import sys
from glob import glob

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def changeMode(mode):
    select = 1
    if mode == "train":
        select = 0
    return f"bool_submission_prep = {str(select)}"


def find_latest_etd(mode):
    feature_dir = "Inference_Extracted_Features"
    if mode == "train":
        feature_dir = "Training_Extracted_Features"

    dirs = glob(f"{feature_dir}/etd_*/")

    dirs_unix = []
    for dir in dirs:
        dir = dir.replace("/", "")
        dirs_unix.append(float(dir.split("_")[-1]))

    max_index = dirs_unix.index(max(dirs_unix))

    newest_dir = dirs[max_index]
    dst_dir = f"{feature_dir}/Current_Features"

    shutil.copytree(newest_dir, dst_dir, dirs_exist_ok=True)


def find_latest_ap_code(mode):
    feature_dir = "Inference_Extracted_Features"
    if mode == "train":
        feature_dir = "Training_Extracted_Features"

    dirs = glob(f"{feature_dir}/airplane_code_*/")

    dirs_unix = []
    for dir in dirs:
        dir = dir.replace("/", "")
        dirs_unix.append(float(dir.split("_")[-1]))

    max_index = dirs_unix.index(max(dirs_unix))

    newest_dir = dirs[max_index]
    dst_dir = f"{feature_dir}/Current_Features"

    shutil.copytree(newest_dir, dst_dir, dirs_exist_ok=True)


def find_latest_tttg(mode):
    feature_dir = "Inference_Extracted_Features"
    if mode == "train":
        feature_dir = "Training_Extracted_Features"

    dirs = glob(f"{feature_dir}/taxitime_to_gate_*/")

    dirs_unix = []
    for dir in dirs:
        dir = dir.replace("/", "")
        dirs_unix.append(float(dir.split("_")[-1]))

    max_index = dirs_unix.index(max(dirs_unix))

    newest_dir = dirs[max_index]
    dst_dir = f"{feature_dir}/Current_Features"

    shutil.copytree(newest_dir, dst_dir, dirs_exist_ok=True)


def find_latest_model():
    dirs = glob("Models/*")
    if os.path.isdir("Models/chosen"):
        dirs.remove("Models/chosen")

    dirs_num = [dir.split("/")[-1] for dir in dirs]

    max_index = dirs_num.index(max(dirs_num))

    found_dir = dirs[max_index]

    shutil.copytree(found_dir, "Models/chosen", dirs_exist_ok=True)


def main(argv):
    ap_code_fp = "Feature_Processing-AirplaneCodes.ipynb"
    etd_no_a_fp = "Feature_Processing-ETD_no_PQDM.ipynb"
    etd_w_a_fp = "Feature_Processing-ETD_PQDM.ipynb"
    tttf_fp = "Feature_Processing-TaxiTimeToGate.ipynb"

    notebooks = [ap_code_fp, tttf_fp]
    a = True
    mode = ""
    opts, args = getopt.getopt(argv, "hdti")
    for opt, arg in opts:
        if opt == "-h":
            print("Options:")
            print("     -h: print options to console")
            print(
                "     -d: if present will opt NOT to use PQDM to accelerate pre-processing of ETD feature"
            )
            print("     -t: Run Training Pipeline")
            print("     -i: Run Inference Pipeline")
            print(
                "**NOTE** if both -t and -i options are present the last argument in the command will be used."
            )
            sys.exit()
        elif opt == "-d":
            a = False
        elif opt == "-t":
            mode = "train"
        elif opt == "-i":
            mode = "infer"

    if a:
        notebooks.append(etd_w_a_fp)
    else:
        notebooks.append(etd_no_a_fp)

    ep = ExecutePreprocessor()
    for notebook in notebooks:
        print(f"-----------------Running {notebook}-----------------")
        with open(notebook) as notebook_file:
            nb = nbformat.read(notebook_file, as_version=4)
            # Make sure we are in proper train or inference mode for preprocessor
            str = nb.cells[1]["source"]
            lines = str.split("\n")
            lines[0] = changeMode(mode)
            new_str = "\n".join(lines)
            nb.cells[1]["source"] = new_str
            ep.preprocess(nb)

    # Find latest save data and copy it over to Current_Features
    dst_dir = "Inference_Extracted_Features/Current_Features/"

    print(
        f"-----------------Copying over newest data to Current_Features to be used for training or inference-----------------"
    )
    find_latest_etd(mode)
    find_latest_ap_code(mode)
    find_latest_tttg(mode)

    infer_fp = "Run_Inference.ipynb"
    train_fp = "Train_Models.ipynb"

    if mode == "train":
        print(f"-----------------Starting Training-----------------")
        with open(train_fp) as notebook_file:
            nb = nbformat.read(notebook_file, as_version=4)
            ep.preprocess(nb)
            find_latest_model()
    else:
        print(f"-----------------Starting Inference----------------")
        with open(infer_fp) as notebook_file:
            nb = nbformat.read(notebook_file, as_version=4)
            ep.preprocess(nb)


if __name__ == "__main__":
    main(sys.argv[1:])
