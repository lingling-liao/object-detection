# Object Detection
Train a object detection model with the TensorFlow Object Detection API.

## Official Docker images for TensorFlow

Docker pull command:

```
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter
```

Running containers:

```
docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter
```

## Install protoc

```
PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

Upgrade pip

```
python -m pip install --upgrade pip
```

## Install Object Detection API (TensorFlow 2)

```
git clone https://github.com/tensorflow/models.git
```

Python Package Installation

```
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

Test the installation.

```
python object_detection/builders/model_builder_tf2_test.py
```

## Install libgl1-mesa-glx for opencv-python

```
apt update
apt install libgl1-mesa-glx
```

## Recommended Directory Structure

```bash
.
├── data/
│   ├── annotations/
│   │   ├── 00001.xml
│   │   ├── 00002.xml
│   │   ├── 00003.xml
│   │   └── ...
|   ├── images/
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   ├── 00003.jpg
│   │   └── ...
│   ├── sets/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   └── label_map.pbtxt
├── fine_tune_checkpoint/
│   ├── checkpoint/
│   │   ├── checkpoint
│   │   ├── ckpt-0.data-00000-of-00001
│   │   └── ckpt-0.index
│   ├── saved_model/
│   │   ├── variables/
│   │   │   ├── variables.data-00000-of-00001
│   │   │   └── variables.index
│   │   └── saved_model.pb
│   └── pipeline.config
└── pipeline.config
```

## Download sample data

https://drive.google.com/drive/folders/1UBVgNKIDbCDlU8-62s0jQzakl5nrFPS2?usp=sharing

## Training a model

Running the command:

```
python train_object_detection_model.py \
    --data_dir=/tf/data \
    --pipeline_config_path=/tf/pipeline.config \
    --model_dir=/tf/saved_model
```

## Object Detection From Checkpoint

```
python train_object_detection_model.py \
    --path_to_ckpt=/tf/saved_model/ckpt-1 \
    --path_to_cfg=/tf/pipeline.config \
    --data_dir=/tf/data \
    --test_set=test \
    --detection_dir=/tf/detection_result
```
