# German-Traffic-Sign-Detection-Benchmark-using-RetinaNet
Training RetinaNet for Traffic Sign detection based on Fizyr implementation of RetinaNet in Keras

Traffic Sign detection palys a significant role in Autonomous driving. This project is to explore the application of RetinaNet on the German Traffic Sign Detection Benchmark Dataset.

The Dataset can be downloaded from http://benchmark.ini.rub.de/?section=gtsdb&subsection=news

Split the ground truth text file `gt.txt` into train, val and test files, define them in a csv format, for more information follow the guide in [keras-retinanet](https://github.com/fizyr/keras-retinanet)

# Keras RetinaNet

RetinaNet as described in this paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) uses a focal loss function training on a sparse set of hard examples and prevents vast number of easy negatives during training.

To install and setup Keras RetinaNet, follow the instructions [here](https://github.com/fizyr/keras-retinanet)


# Dataset

The training and evaluation of this project is based on the German Traffic Sign Detection Benchmark Dataset. It includes images of traffic signs belonging to 43 classes and the data distribution is shown below

![image](/assets/data_dist.png)

Lets have a look at few images of the dataset, the code for visualization can be found in 

![image](/assets/input.png)

## Loading tracklets 

Ground Truth annotations

![image](/assets/tracklets.png)



## Model evaluation

The default backbone of RetinaNet which is resnet50 with pretrained weights on MS COCO dataset was used for transfer learning. The backbone layers were freezed and only top layers were trained.The pretrained MS COCO model can be downloaded [here](https://github.com/fizyr/keras-retinanet/releases). 

![image](/assets/training.JPG)


## Performance 

### Example detection result

raw      |Ground Truth      | RetinaNet 
---      |---------|--------------
![](/assets/raw_1.png)    |![](/assets/tracklet_1.png) | ![](/assets/raw_1.png
![](/assets/raw_2.png)    |![](/assets/tracklet_2.png) | ![](/assets/raw_1.png
![](/assets/raw_3.png)    |![](/assets/tracklet_3.png) | ![](/assets/raw_1.png

### Evaluation metrics

Commonly used metric for object detection is mAP, computed according to the PASCAL VOC dev kit can be found [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)

Results using the `cocoapi` are shown in the paper ( according to the paper, this configuration achieved a mAP of 0.357).
