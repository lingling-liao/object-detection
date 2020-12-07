# object-detection
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
