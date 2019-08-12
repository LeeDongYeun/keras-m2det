# Keras M2det

Keras implementation of M2Det object detection as described in [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/pdf/1811.04533.pdf)
by Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang1, Ying Chen, Ling Cai2 and Haibin Ling.

##Important

The main structure of this project is from [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
Right now, this repository is availble to use just train for vgg16 m2det with CSV datasets
Train the code with

'''
python tkeras_m2det/bin/train.py csv /path/to/custom/csv/data /path/to/custom/class/data
'''

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
5) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb).
In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

Execution time on NVIDIA Pascal Titan X is roughly 75msec for an image of shape `1000x800x3`.

### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## Training
`keras-retinanet` can be trained using [this](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).

Trained models can't be used directly for inference. To convert a trained model to an inference model, check [here](https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model).

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py pascal /path/to/VOCdevkit/VOC2007

# Using the installed script:
retinanet-train pascal /path/to/VOCdevkit/VOC2007
```

For training on [MS COCO](http://cocodataset.org/#home), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py coco /path/to/MS/COCO

# Using the installed script:
retinanet-train coco /path/to/MS/COCO
```

The pretrained MS COCO model can be downloaded [here](https://github.com/fizyr/keras-retinanet/releases). Results using the `cocoapi` are shown below (note: according to the paper, this configuration should achieve a mAP of 0.357).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.350
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.345
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
```

For training on Open Images Dataset [OID](https://storage.googleapis.com/openimages/web/index.html)
or taking place to the [OID challenges](https://storage.googleapis.com/openimages/web/challenge.html), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py oid /path/to/OID

# Using the installed script:
retinanet-train oid /path/to/OID

# You can also specify a list of labels if you want to train on a subset
# by adding the argument 'labels_filter':
keras_retinanet/bin/train.py oid /path/to/OID --labels-filter=Helmet,Tree

# You can also specify a parent label if you want to train on a branch
# from the semantic hierarchical tree (i.e a parent and all children)
(https://storage.googleapis.com/openimages/challenge_2018/bbox_labels_500_hierarchy_visualizer/circle.html)
# by adding the argument 'parent-label':
keras_retinanet/bin/train.py oid /path/to/OID --parent-label=Boat
```


For training on [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py kitti /path/to/KITTI

# Using the installed script:
retinanet-train kitti /path/to/KITTI

If you want to prepare the dataset you can use the following script:
https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/prepare_kitti_data.py
```


For training on a [custom dataset], a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes

# Using the installed script:
retinanet-train csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes
```

In general, the steps to train on your own datasets are:
1) Create a model by calling for instance `keras_retinanet.models.backbone('resnet50').retinanet(num_classes=80)` and compile it.
   Empirically, the following compile arguments have been found to work well:
```python
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.pascal_voc.PascalVocGenerator`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)).
3) Use `model.fit_generator` to start training.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Debugging
Creating your own dataset does not always work out of the box. There is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

## Results

### MS COCO

## Status
Example output images using `keras-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

Contributions to this project are welcome.


