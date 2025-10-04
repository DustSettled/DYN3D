# Dyn3D: Dynamic 3D Scene Reconstruction from Video based on Unified Learning

![image](412.png)
Performing spatiotemporal interpolation of dynamic 3D scenes without effective supervision signals is a critical challenge.Such tasks require models to accurately capture rigid motion trajectories, deformable dynamic details, cross-view structural consistency, and evolutionary patterns across multiple timestamps under the stringent condition of limited strong supervision. However, existing methods often face issues of low reconstruction accuracy, poor perceptual consistency, or weak generalization capability due to their inability to effectively model spatiotemporal correlations.This paper proposes a spatiotemporal interpolation framework for dynamic 3D scenes, named DYN3D, which innovatively introduces a velocity field and a keyframe network. The velocity field learns the motion vector field of objects in the scene to explicitly model their motion trends and dynamic properties, while the keyframe network focuses on mining transition patterns between adjacent keyframes, providing implicit associative guidance for complex motions under sparse timestamp supervision,outperforming both traditional and emerging PINN-based methods.This work provides a reliable solution for high-quality reconstruction of complex dynamic 3D scenes under sparse supervision.

# Environment Setup
```
    # create conda environment
    conda create --name Dyn3D python=3.9
    
    # activate env
    conda activate Dyn3D
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1  cudatoolkit=11.6 -c pytorch -c conda-forge
    pip install functorch==0.2.1
    
    # pip install 
    pip install -r requirements.txt

```
# Datasets
Both Dynamic Object and Dynamic Indoor Scene datasets could be downloaded from [google drive](https://drive.google.com/drive/folders/1je-JW64UvRJ2hmA6nzEKA7VGRIn4lAi6?usp=sharing). 

Please change the "logdir" and "basedir" in config based on the locations of downloaded datasets.

# Training
We provide several config files under [config](./config/) folder for different datasets.

For reconstruction, you can run
```
python train.py --config ./config/InDoorObj/bat.yaml --static_dynamic 
```

# Evaluation
For generating intermediate frames, you can do so by running
```
python train_nvfi.py --config ./config/InDoorObj/bat.yaml --checkpoint -1 --not_train --eval_test
```


