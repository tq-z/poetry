# -*- coding: utf-8 -*-

import preprocess
import tensorflow as tf
import tensorflow.contrib as tf_contrib
import time
import numpy as np
import pickle

word_dict_size = len(preprocess.get_dict())


# def poetry_2_num(poetry):
#     vector = []
#     for word in poetry:
#         vector.append(word_dict.get(word))
#     return vector


class Config(object):
    BATCH_SIZE = 1
    PROB_KEEP = 0.95  # 每此参与训练的节点比例
    HIDEN_SIZE = 1  # 隐藏层神经元个数
    NN_LAYER = 2  # 隐藏层数目
    MAX_GRAD_NORM = 5  # 最大梯度模
    MAX_EPOCH = 50  # 文本循环次数
    LEARNING_RATE = 0.002


# class TrainSet(object):
#     def __init__(self, batch_size, file_path):
#         self._file_path = file_path
#         self._batch_size = batch_size
#         self._poems = []
#         self._poem_vec = []


# batch_times = len(poem_vec) // Config.BATCH_SIZE
with open('./data/Tang.pickle', 'rb') as f:
    x_batches = pickle.load(f)
    y_batches = pickle.load(f)
# with open('./data/Song.pickle', 'rb') as f:
#     x_batches = pickle.load(f)
#     y_batches = pickle.load(f)
data_batche = zip(x_batches, y_batches)

input_ids = tf.placeholder(tf.int32, [Config.BATCH_SIZE, None])
output_targets = tf.placeholder(tf.int32, [Config.BATCH_SIZE, None])


def network(hiden_size=256, layer=2):
    cell_fun = tf.nn.rnn_cell.GRUCell
    cell = cell_fun(hiden_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * layer)
    init_state = cell.zero_state(Config.BATCH_SIZE, tf.float32)

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [word_dict_size, hiden_size])
        inputs = tf.nn.embedding_lookup(embedding, input_ids)
        if Config.PROB_KEEP < 1:  # 这是用来随机扔掉一些不参与训练的
            inputs = tf.nn.dropout(inputs, Config.PROB_KEEP)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state)
    output = tf.reshape(outputs, [-1, hiden_size])

    softmax_w = tf.get_variable("softmax_w", [hiden_size, word_dict_size])  # one-hot表示
    softmax_b = tf.get_variable("softmax_b", [word_dict_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)

    # 计算loss function
    loss = tf_contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits], [tf.reshape(output_targets, [-1])],
        [tf.ones_like(tf.reshape(output_targets, [-1]), dtype=tf.float32)], word_dict_size)  # 交叉熵
    cost = tf.reduce_mean(loss)

    # 算梯度
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), Config.MAX_GRAD_NORM)
    optimizer = tf.train.AdamOptimizer(learning_rate)  #
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    return cost, last_state, train_op, learning_rate, probs, init_state, cell


def train_nn(cost, last_state, op, name, learning_rate):
    start_time = time.time()
    with open('./data/Tang.pickle', 'rb') as ff:
        index = pickle.load(ff)
        epoch = pickle.load(ff)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化
        sess.run(tf.assign(learning_rate, Config.LEARNING_RATE))
        saver = tf.train.Saver(tf.global_variables())   # 保存
        if epoch < Config.MAX_EPOCH:
            epoch += 1
            iters = 0
            costs = 0.0
            if index < len(x_batches):  # 训练一个batch
                index += 1
                train_cost, _, _ = sess.run([cost, last_state, op]
                                            , feed_dict={input_ids: x_batches[index], output_targets: y_batches[index]})
                iters += len(x_batches[index])
                costs += train_cost
                if index % 10 == 0:  # 每10个batch输出1次
                    print('Epoch: %d;batche: %d;loss: %.5f;perplexity: %.3f speed: %.2f' %
                          (epoch, index, train_cost, np.exp(train_cost / iters),
                           iters / (time.time() - start_time)))
                    saver.save(sess, './model/' + name + '.mod', global_step=epoch)  # 保存
                    with open('./model/now.pickle', 'wb') as fff:
                        pickle.dump(index, fff)
                        pickle.dump(epoch, fff)


def creat_poem():
    word_dict = preprocess.get_dict()
    word = word_dict.keys()
    _, last_state, _, _, probs, init_state, cell = network(flag='creat')

    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return word[sample]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './model/test.mod-50')

        state_ = sess.run(cell.zero_state(1, tf.float32))

        x = np.array([list(map(word_dict.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_ids: x, init_state: state_})
        word = to_word(probs_)
        # word = words[np.argmax(probs_)]
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = word_dict[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_ids: x, init_state: state_})
            word = to_word(probs_)
            # word = words[np.argmax(probs_)]
        return poem


if __name__ == '__main__':
    # cost__, last_state__, train_op__, lr, _ = network()
    # train_nn(cost__, last_state__, train_op__, 'test', lr)
    print(creat_poem())


