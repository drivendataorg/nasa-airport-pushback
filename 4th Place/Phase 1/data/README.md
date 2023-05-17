## Data Directory
This repository does not contain the data as it is too large.

Scripts operate under the assumption that there is a directory named "data" in the 
root of the project.

Furthermore, they assume that the data directory has a structure as follows:

```
data
├── <airport>
│   ├── <airport>_config.csv
│   ├── <airport>_etd.csv
│   ├── <airport>_first_position.csv
│   ├── <airport>_lamp.csv
│   ├── <airport>_mfs.csv
│   ├── <airport>_runways.csv
│   ├── <airport>_standtimes.csv
│   ├── <airport>_tbfm.csv
│   └── <airport>_tfm.csv
├── ...
├── train_labels_open
│   ├── train_labels_<airport>.csv
│   └── ...
└──train_labels_prescreened
    ├── prescreened_train_labels_<airport>.csv
    └── ...
```

If it is desired to work with compressed tables to save storage space, the directory should appear as follows:

```
data
├── <airport>
│   ├── <airport>_config.csv.bz2
│   ├── <airport>_etd.csv.bz2
│   ├── <airport>_first_position.csv.bz2
│   ├── <airport>_lamp.csv.bz2
│   ├── <airport>_mfs.csv.bz2
│   ├── <airport>_runways.csv.bz2
│   ├── <airport>_standtimes.csv.bz2
│   └── <airport>_tbfm.csv.bz2
├── ...
├── train_labels_open
│   ├── train_labels_<airport>.csv.bz2
│   └── ...
└──train_labels_prescreened
    ├── prescreened_train_labels_<airport>.csv.bz2
    └── ...
```

The scripts will automatically work with compressed or uncompressed .csv files, including any 
combination of compressed and uncompressed files.
