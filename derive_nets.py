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
layers {
  name: "flatten"
  type: FLATTEN
  bottom: "data"
  top: "flatten"
}
layers {
  name: "fc1"
  type: INNER_PRODUCT
  bottom: "flatten"
  top: "fc1"
  inner_product_param {
    num_output: 100
  }
}
layers {
  name: "fc1_s"
  type: SIGMOID
  bottom: "fc1"
  top: "fc1_s"
}
layers {
  name: "fc2"
  type: INNER_PRODUCT
  bottom: "fc1_s"
  top: "fc2"
  inner_product_param {
    num_output: 100
  }
}
layers {
  name: "fc2_s"
  type: SIGMOID
  bottom: "fc2"
  top: "fc2_s"
}
layers {
  name: "fc2_d"
  type: DROPOUT
  bottom: "fc2_s"
  top: "fc2_sd"
}
layers {
  name: "fc3"
  type: INNER_PRODUCT
  bottom: "fc2_sd"
  top: "fc3"
  inner_product_param {
    num_output: 29
  }
}
'''

train_layers = '''
layers {
  name: "category"
  type: SOFTMAX_LOSS
  bottom: "fc3"
  bottom: "label"
  top: "category"
  include: { phase: TRAIN }
}
layers {
  name: "accuracy"
  type: ACCURACY
  bottom: "fc3"
  bottom: "label"
  top: "accuracy"
  include: { phase: TEST }
}
'''

deploy_layers = '''
layers {
  name: "category"
  type: SOFTMAX
  bottom: "fc3"
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
