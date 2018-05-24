# -*- coding: utf-8 -*-

import preprocess
import tensorflow as tf
import  numpy as np


word_dict = preprocess.creat_dict()


def poetry_2_num(poetry):
    vector = []
    for word in poetry:
        vector.append(word_dict.get(word))
    return vector


BATCH_SIZE = 32


class TrainSet(object):
    def __init__(self, batch_size):
        self._batch_size = batch_size
        


if __name__ == '__main__':
    print(poetry_2_num('我是猪'))
