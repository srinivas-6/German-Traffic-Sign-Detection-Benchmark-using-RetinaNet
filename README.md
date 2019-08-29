# German-Traffic-Sign-Detection-Benchmark-using-RetinaNet
Training RetinaNet for Traffic Sign detection based on Fizyr implementation of RetinaNet in Keras

Traffic Sign detection palys a significant role in Autonomous driving. This project is to explore the application of RetinaNet on the German Traffic Sign Detection Benchmark Dataset.

The Dataset can be downloaded from http://benchmark.ini.rub.de/?section=gtsdb&subsection=news

Split the ground truth text file `gt.txt` into train, val and test files, define them in a csv format, for more information follow the guide in [keras-retinanet](https://github.com/fizyr/keras-retinanet)

# Keras RetinaNet

RetinaNet as described in this paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) uses a focal loss function training on a sparse set of hard examples and prevents vast number of easy negatives during training.

Clone the repository from (https://github.com/fizyr/keras-retinanet) to install and setup Keras RetinaNet, follow the instructions [here](https://github.com/fizyr/keras-retinanet), after you have installed keras-retinanet and defined your data in csv files

There is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes. This tool helps to visualize the annotations and at the same times checks the compatibility of the data

### Training

To train using CSV files, run from the repository:

```shell
# Run directly from the root directory of the cloned repository:
keras_retinanet/bin/train.py csv {PATH TO ANNOTATIONS FILE} {PATH TO CLASSES FILE}
```


# Dataset

Download and extract the dataset, after extracting you will have the following files:
```shell
# images in ppm format
./FulllJCNN2013/00**.ppm

#  ground truth file
./FulllJCNN2013/gt.txt

# Classes - ID mapping
./FulllJCNN2013/classes.txt
```

The training and evaluation of this project is based on the German Traffic Sign Detection Benchmark Dataset. It includes images of traffic signs belonging to 43 classes and the data distribution is shown below

![image](/assets/data_dist.png)

Lets have a look at few images of the dataset, follow the code in [notebook](/data_exploration.ipynb)  for visualization and splitting the data. Make sure that the path specified to load the files are correct. 

![image](/assets/input.png)

## Loading tracklets 

Ground Truth annotations

![image](/assets/tracklets.png)



## Model evaluation

The default backbone of RetinaNet which is resnet50 with pretrained weights on MS COCO dataset was used for transfer learning. The backbone layers were freezed and only top layers were trained.The pretrained MS COCO model can be downloaded [here](https://github.com/fizyr/keras-retinanet/releases). 

After cloning the repo from (https://github.com/fizyr/keras-retinanet) and installing keras-retinanet, run the following from terminal

```shell
# Training keras-retinanet
python keras_retinanet/bin/train.py --weights {PRETRAINED_MODEL} --config ../config.ini --compute-val-loss --weighted-average --multiprocessing --steps 500 --epochs 50 csv ./data/train.csv ./data/classes.csv --val-annotations ./data/val.csv 
```


![image](/assets/training.JPG)


## Performance 

### Example detection result

Input      |RetinaNet output       
---      |---------|
![](/assets/input_1.png)    |![](/assets/output_1.png) 
![](/assets/input_2.png)    |![](/assets/output_2.png) 
![](/assets/input_3.png)    |![](/assets/output_3.png) 
![](/assets/input_4.png)    |![](/assets/output_4.png) 

**Though the model was able to detect the traffic signs in adverse conditions, there were also some misclassifications due to class imbalance as seen in the class distribution graph

Input      |RetinaNet output | Correct label       
---      |---------|------------|
![](/assets/miss_input_1.png)    |![](/assets/miss_output_1.png) |(uneven-road)
![](/assets/miss_input_2.png)    |![](/assets/miss_output_2.png) |(speed-limit 60)
![](/assets/miss_input_3.png)    |![](/assets/miss_output_3.png) |(pedestrian crossing)

### Evaluation metrics

Commonly used metric for object detection is mAP, computed according to the PASCAL VOC dev kit can be found [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)

Results using the `cocoapi` are shown in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) ( according to the paper, this model configuration achieved a mAP of 0.357 on COCO Dataset).

Similary, mAP value was computed while training the model on GTSD dataset. The mAP graph logged in Tensor board shows that the model acheived a maximum of  value 0.30 which is less and main reasons are small dataset size and class imbalance. There are several classes with 0 AP resulting in a low mAP value.

![image](/assets/mAP.PNG)

## TO DO
Data Augmentation to tackle the class imbalance problem 

## Resources
[Traffic-Sign Detection and Classification in the Wild](http://cg.cs.tsinghua.edu.cn/traffic-sign/)

[sridhar912 Traffic Sign Detection and Classification](https://github.com/sridhar912/tsr-py-faster-rcnn)

[Detect and Classify Species of Fish from Fishing Vessels with Modern Object Detectors and Deep Convolutional Networks](https://flyyufelix.github.io/2017/04/16/kaggle-nature-conservancy.html)

## Licenses
German traffic signs detection specific code is distributed under MIT License.
