from typing import Iterable
import tensorflow as tf


def serialize_example(**d):
    feature = {}
    for key, value in d.items():
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        dtype = str(value.dtype)
        if not isinstance(value, Iterable):
            value = [value]
        if dtype in {'float16', 'float32', 'float64'}:
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        elif dtype in {'int16', 'int32', 'int64'}:
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        else:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()