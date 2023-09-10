### My Setup Environment

* docker pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel (https://hub.docker.com/layers/pytorch/pytorch/1.13.1-cuda11.6-cudnn8-devel/images/sha256-58d848c38665fd3ed20bee65918255cb083637c860eb4fae67face2fb2ff5702?context=explore) 
* pip install -U scikit-learn scipy matplotlib
* pip install pandas
* pip install tensorboardX
* pip install opencv-python
* apt-get install libgl1

### Datasets folder structure

    .
    ├── ...
    ├── datasets                    
    │   ├── oz_test
    |       ├── casia_faces
    |       ├── lfw             
    └── ...

### Run Tensorboard

    pip install tensorboard
    cd log && tensorboard --logdir=./

### PS

Second commit includes all my changes to the 'face.evoLVe' repository.