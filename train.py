# -*- coding: utf-8 -*-

import preprocess
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import time
import numpy as np

word_dict = preprocess.get_dict()

# def poetry_2_num(poetry):
#     vector = []
#     for word in poetry:
#         vector.append(word_dict.get(word))
#     return vector


class Config(object):
    BATCH_SIZE = 77
    PROB_KEEP = 1.0  # 每此参与训练的节点比例
    HIDEN_SIZE = 256  # 隐藏层神经元个数
    NN_LAYER = 3  # 隐藏层数目
    MAX_GRAD_NORM = 5  # 最大梯度模
    MAX_EPOCH = 30  # 文本循环次数


class TrainSet(object):
    def __init__(self, batch_size, file_path):
        self._file_path = file_path
        self._batch_size = batch_size
        self._poems = []
        self._poem_vec = []


batch_times = len(poem_vec) // Config.BATCH_SIZE
x_batches, y_batches = [], []
data_batche = zip(x_batches, y_batches)

input_ids = tf.placeholder(tf.int16, [Config.BATCH_SIZE, None])
output_targets = tf.placeholder(tf.int16, [Config.BATCH_SIZE, None])

def network(hiden_size=256, layer=3):
    cell_fun = tf.nn.rnn_cell.GRUCell
    cell = cell_fun(hiden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer)
    init_state = cell.zero_state(Config.BATCH_SIZE, tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [len(word_dict) + 1, hiden_size])
        inputs = tf.nn.embedding_lookup(embedding, input_ids)
        if Config.PROB_KEEP < 1:  # 这是用来随机扔掉一些不参与训练的
            inputs = tf.nn.dropout(inputs, Config.PROB_KEEP)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
    output = tf.reshape(outputs, [-1, hiden_size])

    softmax_w = tf.get_variable("softmax_w", [hiden_size, len(word_dict) + 1])  # one-hot表示
    softmax_b = tf.get_variable("softmax_b", [len(word_dict) + 1])
    logits = tf.matmul(output, softmax_w) + softmax_b

    # 计算loss function
    loss = tf_contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits], [tf.reshape(output_targets, [-1])],
        [tf.ones([hiden_size * word_dict], dtype=tf.float16)],
    )  # 交叉熵
    cost = tf.reduce_mean(loss)

    # 算梯度
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_average_norm(tf.gradients(cost, tvars), Config.MAX_GRAD_NORM)
    optimizer = tf.train.AdamOptimizer(learning_rate)  #
    train_op = optimizer.apply_gradients(zip(grads, tvars))
    return cost, last_state, train_op


def train_nn(seesion, cost, last_state, op, name):
    start_time = time.time()
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())  # 初始化
        saver = tf.train.Saver(tf.all_variables())  # 保存
        for epoch in range(Config.MAX_EPOCH):
            iters = 0
            costs = 0.0
            for index, (x, y) in enumerate(data_batche):
                #训练一个batch
                train_cost, _, _ = sess.run([cost, last_state, op]
                                            , feed_dict={input_ids: x, output_targets: y})
                iters += len(x)
                costs += train_cost
                if index % 10 == 0:  # 每10个batch输出1次
                    print('Epoch: %d;batche: %d;loss: %.5f;perplexity: %.3f speed: %.2f' %
                          (epoch, index, train_cost, np.exp(train_cost / iters),
                           iters / (time.time() - start_time)))
                    saver.save(sess, name + '.mod', global_step=epoch)#保存


# def creat_poem()


if __name__ == '__main__':
    print(poetry_2_num('我是猪'))
