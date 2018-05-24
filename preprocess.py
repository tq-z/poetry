# -*- coding: utf-8 -*-

#   把全宋词处理的和全唐诗一样
#   格式  名:词

import collections as coll


def preprocess_song():
    title = []
    line_list = []
    flag = 0
    with open('./data/Song.txt', 'w', encoding='utf-8') as fw:
        with open('./data/QuanSong.txt', 'r', encoding='utf-8') as fr:
            for line in fr.readlines():

                print(title, flag)
                print(len(line), line)

                if len(line) >= 18:
                    flag += 1
                    line_list += [line[0:-1]]
                    print(line_list)

                elif (len(line) < 18) and (flag != 0):
                    fw.writelines(title)
                    for lines in line_list:
                        fw.writelines(line_list)
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
            for item in word[0]:
                words.append(item)
    words = list(set(words))
    with open('./data/Tang.txt', 'r', encoding='utf-8') as fr:
        for word in fr:
            for item in word[0]:
                words.append(item)
    words = list(set(words))
    # print(words)
    word_dict = dict(zip(words, range(len(words))))
    with open('./data/Dict.txt', 'w', encoding='utf-8') as fw:
        fw.write(str(word_dict))
    return word_dict


if __name__ == '__main__':
    creat_dict()
    with open('./data/Dict.txt', 'r', encoding='utf-8') as f:
        dict_w = eval(f.read())
        print(dict_w)