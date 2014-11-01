#!/bin/python3

train_inputs = '''
layers {
  name: "data_source"
  type: IMAGE_DATA
  top: "data"
  top: "label"
  image_data_param {
    source: "derived/cells/train.txt"
    batch_size: 32
    new_width: 80
    new_height: 80
  }
  transform_param {
    scale: 0.0039215684
  }
  include { phase: TRAIN }
}
layers {
  name: "data_source"
  type: IMAGE_DATA
  top: "data"
  top: "label"
  image_data_param {
    source: "derived/cells/test.txt"
    batch_size: 32
    new_width: 80
    new_height: 80
  }
  transform_param {
    scale: 0.0039215684
  }
  include { phase: TEST }
}
'''

deploy_inputs = '''
input: "data"
input_dim: 1
input_dim: 3
input_dim: 80
input_dim: 80
'''

common_layers = '''
name: "cells-simple"
# convolution - relu - max pool
layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 15
    kernel_size: 5
    stride: 4
    weight_filler {
      type: "xavier"
    }
  }
}
layers {
  name: "relu1"
  type: RELU
  bottom: "conv1"
  top: "relu1"
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "relu1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
# convolution - relu - max pool
layers {
  name: "conv2"
  type: CONVOLUTION
  bottom: "pool1"
  top: "conv2"
  convolution_param {
    num_output: 15
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layers {
  name: "relu2"
  type: RELU
  bottom: "conv2"
  top: "relu2"
}
layers {
  name: "pool2"
  type: POOLING
  bottom: "relu2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
# fully connected layers
layers {
  name: "fc1"
  type: INNER_PRODUCT
  bottom: "pool2"
  top: "fc1"
  inner_product_param {
    num_output: 100
    weight_filler {
      type: "xavier"
    }
  }
}
layers {
  name: "fc1_s"
  type: RELU
  bottom: "fc1"
  top: "fc1_s"
}
layers {
  name: "fc1_d"
  type: DROPOUT
  bottom: "fc1_s"
  top: "fc1_sd"
}
layers {
  name: "fc2"
  type: INNER_PRODUCT
  bottom: "fc1_sd"
  top: "fc2"
  inner_product_param {
    num_output: 29
    weight_filler {
      type: "xavier"
    }
  }
}
'''

train_layers = '''
layers {
  name: "category"
  type: SOFTMAX_LOSS
  bottom: "fc2"
  bottom: "label"
  top: "category"
  include: { phase: TRAIN }
}
layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "fc2"
  bottom: "label"
  top: "accuracy"
  include: { phase: TEST }
}
'''

deploy_layers = '''
layers {
  name: "category"
  type: SOFTMAX
  bottom: "fc2"
  top: "category"
}
'''

if __name__ == '__main__':
    with open('cells-net-train.prototxt', 'w') as f:
        f.write(train_inputs)
        f.write(common_layers)
        f.write(train_layers)
    with open('cells-net-deploy.prototxt', 'w') as f:
        f.write(deploy_inputs)
        f.write(common_layers)
        f.write(deploy_layers)
