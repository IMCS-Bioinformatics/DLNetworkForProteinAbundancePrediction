# Distributed under The MIT License (MIT)
# Copyright (c) 2017 Karlis Freivalds

import tensorflow as tf

def variance_scaling_lr(shape, mode='FAN_IN'):
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
      raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)
    if mode == 'FAN_IN':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'FAN_OUT':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'FAN_AVG':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
    trunc_stddev = tf.sqrt(1.3 / n)
    return trunc_stddev


def conv_eq(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                scope = "conv2d",
                activation_fn=None,
                normalizer_fn=None,
                kernel_mult = 1.0,
                init_zero = False,
                use_noise = False
            ):
    with tf.variable_scope(scope):
        in_shape = inputs.get_shape().as_list()
        in_dims = in_shape[-1]
        kernel_shape = kernel_size+[in_dims, num_outputs]
        kernel_scale = variance_scaling_lr(kernel_shape, mode = 'FAN_AVG')
        if init_zero:
            kernel = tf.get_variable("kernel", initializer = tf.zeros(kernel_shape))
        else:
            kernel = tf.get_variable("kernel", initializer=tf.truncated_normal(kernel_shape, 0.0, 1))
        kernel*=kernel_scale*kernel_mult

        p_len_0 = (kernel_size[0]-1)//2
        p_len_1 = (kernel_size[1] - 1) // 2
        inputs = tf.pad(inputs, [[0, 0], [p_len_0, p_len_1], [p_len_0, p_len_1], [0, 0]], 'REFLECT')
        res = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding='VALID')

        if normalizer_fn is None:
            #if not init_zero:
                bias = tf.get_variable("bias", [1, 1, 1, num_outputs], initializer=tf.constant_initializer(0.0))
                res+=bias
        else:
            res = normalizer_fn(res)

        if use_noise: res += tf.truncated_normal(tf.shape(res), stddev=0.01*kernel_mult)

        if activation_fn is not None:
            res = activation_fn(res)

        return res

def linear_eq(input_, output_size, scope=None,
              activation_fn=None,
                normalizer_fn=None,
                kernel_mult = 1.0,use_noise = False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        kernel_shape = [shape[1], output_size]
        matrix = tf.get_variable("Matrix", kernel_shape, tf.float32, tf.truncated_normal_initializer(stddev=1.0))
        scale = variance_scaling_lr(kernel_shape, mode='FAN_AVG')
        matrix *= scale*kernel_mult

        res = tf.matmul(input_, matrix)

        if normalizer_fn is None:
            bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
            res+=bias
        else:
            res = normalizer_fn(res)

        if use_noise: res += tf.truncated_normal(tf.shape(res), stddev=0.01*kernel_mult)

        if activation_fn is not None:
            res = activation_fn(res)

        return res
