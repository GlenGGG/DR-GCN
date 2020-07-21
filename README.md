# DR-GCN
Dyadic Relational Graph Convolutional Network (DR-GCN) for skeleton based interaction recognition.



# Data Preparation

 - Download skeleton data of NTU-RGB+D and NTU-RGB+D 120 from their [official website](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp). Then put them under the data directory:

        -data\
          -nturgbd_raw\
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
	   -nturgbd120_raw\
            -nturgb+d120_skeletons\
              ...
            -samples_with_missing_skeletons.txt

 - Preprocess the data with

    `python data_gen/ntu_gendata.py`

    `python data_gen/ntu120_gendata.py.`

 - Generate the bone data with:

    `python data_gen/gen_bone_data.py`

# Training & Testing

Change the config file depending on what you want. Note, configs with "dr_" prefix are for our 2s-DRAGCN, those without this prefix are for 2s-AGCN.


    `python main.py --config ./config/nturgbd-cross-view/dr_train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/dr_train_bone.yaml`
To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer.

    `python main.py --config ./config/nturgbd-cross-view/dr_test_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/dr_test_bone.yaml`

Then combine the generated scores with:

    `python ensemble.py` --datasets ntu/xview --models dragcn
