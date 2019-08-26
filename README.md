# German-Traffic-Sign-Detection-Benchmark-using-RetinaNet
Training RetinaNet for Traffic Sign detection based on Fizyr implementation of RetinaNet in Keras

Traffic Sign detection palys a significant role in Autonomous driving. This project is to explore the application of RetinaNet on the German Traffic Sign Detection Benchmark Dataset.

The Dataset can be downloaded from http://benchmark.ini.rub.de/?section=gtsdb&subsection=news

# Keras RetinaNet

RetinaNet as described in this paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) uses a focal loss function training on a sparse set of hard examples and prevents vast number of easy negatives during training.

To install and setup Keras RetinaNet, follow the instructions [here] (https://github.com/fizyr/keras-retinanet)


# Dataset

The training and evaluation of this project is based on the German Traffic Sign Detection Benchmark Dataset. It includes images of traffic signs belonging to 43 classes and the data distribution is shown below


