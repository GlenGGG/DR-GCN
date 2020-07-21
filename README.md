# DR-GCN
Dyadic Relational Graph Convolutional Network (DR-GCN) for skeleton based interaction recognition.

Two Stream Dyadic Relational AGCN (2s-DRAGCN) in our paper can be found in [here](https://github.com/GlenGGG/2s-DRAGCN).

This repository is based on [mmskeleton](https://github.com/open-mmlab/mmskeleton).

![An illustration of DR-GCN's model sturcture.](/resource/pic/structure.jpg)

# Install mmskeleton

 - mmskeleton supports only Linux systems
 
 - Setup:
 
 `python setup.py develop`
 
 - If anything went wrong, [follow their guide](https://github.com/open-mmlab/mmskeleton/blob/master/doc/GETTING_STARTED.md) for the installation.

# Data Preparation

 - Download skeleton data of NTU-RGB+D and NTU-RGB+D 120 from their [official website](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). Download SBU dataset from their [official website](http://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html).

 - Put data of NTU-RGB+D and NTU-RGB+D 120 under the data directory:

        -data\
          -nturgbd_raw\
            -nturgb+d_skeletons\
              ...
            -nturgbd_samples_with_missing_skeletons.txt
	   -nturgbd120_raw\
            -nturgb+d120_skeletons\
              ...
            -nturgbd120_samples_with_missing_skeletons.txt

  - Put data of SBU under the data directory:

        -data\
	    -sbu_raw\
	     -Set01
	     -Set02
	      ...
	     -Set21

 - Preprocess the data with

```
    `python data_processing/ntu_gendata.py --data_path <path to nturgbd+d_skeletons>`

    `python data_processing/ntu120_gendata.py --data_path <path to nturgbd+d120_skeletons>`
    
    `python data_processing/sbu_gendata.py --data_path <pth to sbu_raw>`
```

# Training & Testing

Change the config file depending on what you want. 

Note, the "duo_only" option in config files are solely used for the extraction of interaction classes from NTU-RGB+D and NTU-RGB+D 120 datasets. This extraction is done in [mmskeleton/feeder/skeleton_feeder.py](/mmskeleton/feeder/skeleton_feeder.py). If you have extracted interaction classes yourself, turn this off by setting "duo_only: False" in config files. Also, do not use this option for SBU dataset.

    `mmskl configs/recognition/dr_gcn/$DATASET/train.yaml`
    
    `$DATASET can be:`
		`ntu120-rgbd-xset`
		`ntu120-rgbd-xsub`
		`ntu-rgbd-xsub`
		`ntu-rgbd-xview`
		`SBU`

