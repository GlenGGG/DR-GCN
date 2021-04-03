# DR-GCN
This is the official repository for _Dyadic Relational Graph Convolutional Network (DR-GCN) for skeleton based interaction recognition_ published in Pattern Recognition. [Paper Link](https://www.sciencedirect.com/science/article/pii/S0031320321001072).

Two Stream Dyadic Relational AGCN (2s-DRAGCN) in [our paper](https://www.sciencedirect.com/science/article/pii/S0031320321001072) can be found in [here](https://github.com/GlenGGG/2s-DRAGCN).

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

# Citation
BibTex:
```
@article{ZHU2021107920,
title = {Dyadic relational graph convolutional networks for skeleton-based human interaction recognition},
journal = {Pattern Recognition},
volume = {115},
pages = {107920},
year = {2021},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2021.107920},
url = {https://www.sciencedirect.com/science/article/pii/S0031320321001072},
author = {Liping Zhu and Bohua Wan and Chengyang Li and Gangyi Tian and Yi Hou and Kun Yuan},
keywords = {3D skeleton-based interaction recognition, Multi-scale graph convolution networks, Graph inference},
abstract = {Skeleton-based human interaction recognition is a challenging task requiring all abilities to recognize spatial, temporal, and interactive features. These abilities rarely co-exist in existing methods. Graph convolutional network (GCN) based methods fail to extract interactive features. Traditional interaction recognition methods cannot effectively capture spatial features from skeletons. Toward this end, we propose a novel Dyadic Relational Graph Convolutional Network (DR-GCN) for interaction recognition. Specifically, we make four contributions: (i) we design a Relational Adjacency Matrix (RAM) that represents dynamic relational graphs. These graphs are constructed combining both geometric features and relative attention from the two skeleton sequences; (ii) we propose a Dyadic Relational Graph Convolution Block (DR-GCB) that extracts spatial-temporal interactive features; (iii) we stack the proposed DR-GCBs to build DR-GCN and integrate our methods with an advanced model. (iv) Our models achieve state-of-the-art results on SBU and significant improvements on the mutual action sub-datasets of NTU-RGB+D and NTU-RGB+D 120.}
}
```
