# DYN3D
DYN3D: UNIFIED LEARNING OF DYNAMIC 3D SCENES FROM VIDEO

Datasets
Both Dynamic Object and Dynamic Indoor Scene datasets could be downloaded from google drive or from HuggingFace: Dynamic Objects, Dynamic Indoor Scenes. We split the data in HuggingFace version for easier evaluation, please follow the datacard there.

Please change the "logdir" and "basedir" in config based on the locations of downloaded datasets.

Training
We provide several config files under config folder for different datasets.

For reconstruction, you can run

python train_nvfi.py --config ./config/InDoorObj/bat.yaml --static_dynamic 
If you want to train the segmentation fields, simply run the following commend

python train_segm.py --config config/InDoorObj/bat.yaml --checkpoint -1
Evaluation
For future frame extrapolation, you can render the results by running

python train_nvfi.py --config ./config/InDoorObj/bat.yaml --checkpoint -1 --not_train --eval_test
For segmentation prediction, you can run

python test_segm_render.py --config config/InDoorObj/bat.yaml --checkpoint -1 --ckpt_segm 1000
