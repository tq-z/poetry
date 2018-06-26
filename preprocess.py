# -*- coding: utf-8 -*-

#   把全宋词处理的和全唐诗一样
#   格式  名:词

import collections as coll
import numpy as np
import pickle


class Config(object):
    BATCH_SIZE = 77
    PROB_KEEP = 0.95  # 每此参与训练的节点比例
    HIDEN_SIZE = 256  # 隐藏层神经元个数
    NN_LAYER = 3  # 隐藏层数目
    MAX_GRAD_NORM = 5  # 最大梯度模
    MAX_EPOCH = 50  # 文本循环次数


def preprocess_song():
    title = []
    line_list = []
    flag = 0
    with open('./data/Song.txt', 'w', encoding='utf-8') as fw:
        with open('./data/QuanSong.txt', 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                # print(title, flag)
                # print(len(line), line)
                if len(line) >= 18:
                    flag += 1
                    line_list += [line[0:-1]]
                    # print(line_list)

                elif (len(line) < 18) and (flag != 0):
                    fw.writelines(title)
                    for lines in line_list:
                        fw.writelines(lines)
                    fw.write('\n')
                    title = [line[0:-1], ':']
                    flag = 0
                    line_list = []
                else:
                    title = [line[0:-1], ':']
                    flag = 0

                # line = fr.readline()


def creat_dict():
    words = []
    with open('./data/Song.txt', 'r', encoding='utf-8') as fr:
        for word in fr:
            for item in word:
                words.append(item)
    words = list(set(words))
    with open('./data/Tang.txt', 'r', encoding='utf-8') as fr:
        for word in fr:
            for item in word:
                words.append(item)
    words = list(set(words))
    # print(words)
    _word_dict = dict(zip(words, range(len(words))))
    with open('./data/Dict.txt', 'w', encoding='utf-8') as fw:
        fw.write(str(_word_dict))
    return _word_dict


def get_dict():
    with open('./data/Dict.txt', 'r', encoding='utf-8') as f:
        dict_w = eval(f.read())
    return dict_w


def poterys_2_num_file(file_path, ddict):
    def poetry_2_num(poetry):
        vector = []
        for word in poetry:
            vector.append(ddict.get(word))
        return vector

    poterys = []
    x_batches = []
    y_batches = []
    with open(file_path, "r", encoding='utf-8', ) as fr:
        for line in fr.readlines():
            title, content = line.strip().split(':')
            content = '[' + content.replace(' ', '') + ']'
            poterys.append(poetry_2_num(content))
        poterys = sorted(poterys, key=lambda i: len(i))
    for index in range(len(poterys)):
        if index % Config.BATCH_SIZE == 0:
            begin_index = index
            end_index = begin_index + Config.BATCH_SIZE
            temp = poterys[begin_index:end_index]  # 从诗list中选取一个batch
            length = max(map(len, temp))  # 这个batch中最长的诗的长度
            x_one = np.full((Config.BATCH_SIZE, length), ddict[' '], np.int16)
            for row in range(len(temp)):
                # 读取一行诗
                x_one[row, :len(temp[row])] = temp[row]
            y_one = np.copy(x_one)
            y_one[:, :-1] = x_one[:, 1:]  # y = x >> 1
            x_batches.append(x_one)
            y_batches.append(y_one)

    # print(x_batches)
    # np.save('./data/shit.npy', np.array(x_batches))
    # np.savez('.'+file_path.split('.')[1] + '.npz', xx_batches=x_batches, yy_batches=y_batches)
    with open('.' + file_path.split('.')[1] + '.pickle', 'wb') as f:
        pickle.dump(x_batches, f)
        pickle.dump(y_batches, f)


if __name__ == '__main__':
    # preprocess_song()
    # creat_dict()
    # word_dict = get_dict()
    # poterys_2_num_file('./data/Tang.txt', word_dict)
    # poterys_2_num_file('./data/Song.txt', word_dict)
    # with open('./data/Tang.pickle', 'rb') as f:
    #     xx = pickle.load(f)
    #     yy = pickle.load(f)

    a = get_dict()