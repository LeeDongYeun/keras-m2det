"""
Copyright 2019 LeeDongYeun (https://github.com/LeeDongYeun/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import numpy as np
import tensorflow as tf
import keras_resnet

from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def upsample_add(tensors):
    _, h, w, _ = tensors[1].shape

    h = int(h)
    w = int(w)
    up = tf.image.resize_bilinear(tensors[0], size=(h, w))
    out = up + tensors[1]

    return out


def upsample_add_output_shape(input_shapes):
    shape = list(input_shapes[1])

    return shape


def Conv(inputs, filters, kernel_size, strides, padding, name='conv'):

    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_conv')(inputs)
    bn = keras_resnet.layers.BatchNormalization(freeze=False, name=name+'_BN')(conv)
    relu = keras.layers.ReLU(name=name)(bn)

    return relu


def FFMv1(input_size_1=(40, 40, 512), input_size_2=(20, 20, 512), feature_size_1=256, feature_size_2=512,
          name='FFMv1'):
    C4 = keras.layers.Input(input_size_1)
    C5 = keras.layers.Input(input_size_2)

    F4 = Conv(C4, filters=feature_size_1, kernel_size=(3, 3), strides=(1, 1), padding='same', name='F4')

    F5 = Conv(C5, filters=feature_size_2, kernel_size=(1, 1), strides=(1, 1), padding='same', name='F5')
    F5 = keras.layers.UpSampling2D(size=(2, 2), name='F5_Up')(F5)

    outputs = keras.layers.Concatenate(name=name)([F4, F5])

    return keras.models.Model(inputs=[C4, C5], outputs=outputs, name=name)


def FFMv2(stage, base_size=(40,40,768), tum_size=(40,40,128), feature_size=128, name='FFMv2'):
    base = keras.layers.Input(base_size)
    tum = keras.layers.Input(tum_size)

    outputs = Conv(base, filters=feature_size, kernel_size=(1, 1), strides=(1, 1), padding='same', name=name+"_"+str(stage) + '_base_feature')
    outputs = keras.layers.Concatenate(name=name+"_"+str(stage))([outputs, tum])

    return keras.models.Model(inputs=[base, tum], outputs=outputs, name=name+"_"+str(stage))


def TUM(stage, input_size=(40, 40, 256), feature_size=256, name="TUM"):
    output_features = feature_size // 2

    conv_TUM_name_base = 'conv_' + str(stage) + '_level'
    size_buffer = []

    inputs = keras.layers.Input(input_size)
    # level one
    level = 1
    f1 = inputs
    f2 = Conv(f1, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f2')
    f3 = Conv(f2, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f3')
    f4 = Conv(f3, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f4')
    f5 = Conv(f4, filters=feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same',name=name + "_" + str(stage) + '_f5')
    f6 = Conv(f5, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_f6')

    size_buffer.append([int(f1.shape[2])] * 2)
    size_buffer.append([int(f2.shape[2])] * 2)
    size_buffer.append([int(f3.shape[2])] * 2)
    size_buffer.append([int(f4.shape[2])] * 2)
    size_buffer.append([int(f5.shape[2])] * 2)
    size_buffer.append([int(f6.shape[2])] * 2)

    # print(size_buffer)

    # level two:using Blinear Upsample + ele-wise sum
    # define a Lambda function to compute upsample_blinear
    level = 2
    c6 = f6

    c5 = Conv(c6, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same',name=name + "_" + str(stage) + '_c5')
    c5 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[4]), name=name + "_" + str(stage) + '_upsample_add5')(c5)
    c5 = keras.layers.Add()([c5, f5])
    # c5 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add5')([c5, f5])

    c4 = Conv(c5, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c4')
    c4 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[3]), name=name + "_" + str(stage) + '_upsample_add4')(c4)
    c4 = keras.layers.Add()([c4, f4])
    # c4 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add4')([c4, f4])

    c3 = Conv(c4, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c3')
    c3 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[2]), name=name + "_" + str(stage) + '_upsample_add3')(c3)
    c3 = keras.layers.Add()([c3, f3])
    # c3 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add3')([c3, f3])

    c2 = Conv(c3, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c2')
    c2 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[1]), name=name + "_" + str(stage) + '_upsample_add2')(c2)
    c2 = keras.layers.Add()([c2, f2])
    # c2 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add2')([c2, f2])

    c1 = Conv(c2, filters=feature_size, kernel_size=(3, 3), strides=(1, 1), padding='same', name=name + "_" + str(stage) + '_c1')
    c1 = keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size=size_buffer[0]), name=name + "_" + str(stage) + '_upsample_add1')(c1)
    c1 = keras.layers.Add()([c1, f1])
    # c1 = keras.layers.Lambda(upsample_add, upsample_add_output_shape, name=name + "_" + str(stage) + '_upsample_add1')([c1, f1])

    # level three:using 1 * 1 kernel to make it smooth
    level = 3

    o1 = Conv(c1, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o1')
    o2 = Conv(c2, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o2')
    o3 = Conv(c3, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o3')
    o4 = Conv(c4, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o4')
    o5 = Conv(c5, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o5')
    o6 = Conv(c6, filters=output_features, kernel_size=(1, 1), strides=(1, 1), padding='valid',name=name + "_" + str(stage) + '_o6')

    outputs = [o1, o2, o3, o4, o5, o6]

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name + "_" + str(stage))


def _concatenate_features(features):
    transposed = np.array(features).T
    transposed = np.flip(transposed, 0)

    concatenate_features = []
    for features in transposed:
        concat = keras.layers.Concatenate()([f for f in features])
        concatenate_features.append(concat)

    return concatenate_features


def _create_feature_pyramid(base_feature, stage=6):
    features = []

    inputs = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(base_feature)
    tum = TUM(1)

    outputs = tum(inputs)
    max_output = outputs[0]
    features.append(outputs)

    for i in range(2, stage+1):
        ffmv2 = FFMv2(i - 1)
        inputs = ffmv2([base_feature, max_output])

        tum = TUM(i)
        outputs = tum(inputs)

        max_output = outputs[0]
        features.append(outputs)

    feature_pyramid = _concatenate_features(features)

    print(tum.summary())
    print(ffmv2.summary())

    return feature_pyramid


def _calculate_input_sizes(concatenate_features):
    input_size = []
    for features in concatenate_features:
        size = (int(features.shape[1]), int(features.shape[2]), int(features.shape[3]))
        input_size.append(size)

    return input_size


def SE_block(input_size, compress_ratio=16, name='SE_block'):
    inputs = keras.layers.Input(input_size)

    pool = keras.layers.GlobalAveragePooling2D()(inputs)
    reshape = keras.layers.Reshape((1, 1, input_size[2]))(pool)
    fc1 = keras.layers.Conv2D(filters=input_size[2] // compress_ratio, kernel_size=1, strides=1, padding='valid',
                              activation='relu', name=name+'_fc1')(reshape)
    fc2 = keras.layers.Conv2D(filters=input_size[2], kernel_size=1, strides=1, padding='valid', activation='sigmoid',
                              name=name+'_fc2')(fc1)

    reweight = keras.layers.Multiply(name=name+'_reweight')([inputs, fc2])

    return keras.models.Model(inputs=inputs, outputs=reweight, name=name)


def SFAM(input_sizes, compress_ratio=16, name='SFAM'):
    inputs = []
    outputs = []
    for i in range(len(input_sizes)):
        input_size = input_sizes[i]
        _input = keras.layers.Input(input_size)

        se_block = SE_block(input_size, compress_ratio=compress_ratio, name='SE_block_' + str(i))

        _output = se_block(_input)

        inputs.append(_input)
        outputs.append(_output)

    print(se_block.summary())

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    print(model.summary())

    return model


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs, name=name)
    print(model.summary())

    return model


def default_submodels(num_classes, num_anchors):
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model(4, num_anchors, pyramid_feature_size=1024)),
        ('classification', default_classification_model(num_classes, num_anchors, pyramid_feature_size=1024))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def m2det(inputs, backbone_layers, num_classes, num_anchors=None, submodels=None, name='m2det'):
    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors)

    C3, C4, C5 = backbone_layers
    _, h4, w4, f4 = C4.shape
    _, h5, w5, f5 = C5.shape

    C4_shape = (int(h4), int(w4), int(f4))
    C5_shape = (int(h5), int(w5), int(f5))

    ffmv1 = FFMv1(C4_shape, C5_shape, feature_size_1=256, feature_size_2=512)
    print(ffmv1.summary())
    base_feature = ffmv1([C4, C5])

    feature_pyramid = _create_feature_pyramid(base_feature, stage=8)
    feature_pyramid_sizes = _calculate_input_sizes(feature_pyramid)

    sfam = SFAM(feature_pyramid_sizes)
    print(sfam.summary())
    outputs = sfam(feature_pyramid)

    pyramids = __build_pyramid(submodels, outputs)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def m2det_bbox(model=None, nms=True, class_specific_filter=True, name='m2det-bbox', anchor_params=None, **kwargs):
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create m2det model
    if model is None:
        model = m2det(num_anchors=anchor_params.num_anchors(), **kwargs)
    else:
        assert_training_model(model)

    feature_layer = model.get_layer("SFAM")
    features = feature_layer.get_output_at(1)
    anchors = __build_anchors(anchor_params, features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])
    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification] + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)