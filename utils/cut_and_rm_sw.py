# coding=utf-8
import os

import jieba

jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/user_words.txt')
import jieba.analyse
import jieba.posseg as pseg
import re


def get_stopwords(base_dir: str = '../dicts/'):
    sw_set = set()

    with open(base_dir + '/baidu_stopwords.txt') as f:
        [sw_set.add(line.strip()) for line in f.readlines()]

    with open(base_dir + '/cn_stopwords.txt') as f:
        [sw_set.add(line.strip()) for line in f.readlines()]

    with open(base_dir + '/hit_stopwords.txt') as f:
        [sw_set.add(line.strip()) for line in f.readlines()]

    with open(base_dir + '/scu_stopwords.txt') as f:
        [sw_set.add(line.strip()) for line in f.readlines()]

    return sw_set


# 词性标注，nr为人名
def get_first_name(content):
    words = pseg.cut(content)
    for word, flag in words:
        if flag == 'nr' and len(word) > 1:  # 单字姓名去掉
            return word

    return False


def get_all_Name(content):
    words = pseg.cut(content)
    names = []
    for word, flag in words:
        # print('%s,%s' % (word, flag))
        if flag == 'nr':  # 人名词性为nr
            names.append(word)
    return names


def cut_sentence(can: str = '', drop_threshold=10, sw_set: set = None):
    words = pseg.cut(can)

    cutted_words = []
    for word, flag in words:
        # ['a', 'v', 'n', 'an', 'vn', 'nz', 'nt', 'nr', ]
        # 'v', 'n', 'ns', 't', 'a', 'vn', 's', 'b'
        if (flag in ['v', 'n', 'a', 'vn']) and (word not in sw_set) and (
                not re.match('^[0-9|.]*$', word)) and (len(word) > 1):
            cutted_words.append(word)

    if len(cutted_words) < drop_threshold:
        return "", 0

    return " ".join(cutted_words), len(cutted_words)


def cut_sentence_without_flag(can: str = '', drop_threshold=10, sw_set: set = None):
    words = jieba.cut(can)

    cutted_words = []
    for word in words:
        if (word not in sw_set) and (not re.match('^[0-9|.]*$', word)) and (len(word) > 1):
            cutted_words.append(word)

    if len(cutted_words) < drop_threshold:
        return "", 0

    return " ".join(cutted_words), len(cutted_words)


def cut_documents(src_file_name: str, des_file_name: str, sw_set: set, with_tag=True, separator='\r'):
    minl = 1000
    maxl = 0
    avg = 0
    cnt = 0
    jieba.enable_parallel(10)
    if os.path.exists(des_file_name):
        os.remove(des_file_name)
    with open(src_file_name, 'r') as f:
        while True:
            can = f.readline()
            if not can:
                break
            if can.strip() == '':
                continue

            cutted_sentence, length = cut_sentence(can, drop_threshold=10, sw_set=sw_set) if with_tag else cut_sentence_without_flag(can, drop_threshold=10, sw_set=sw_set)

            if length < minl:
                minl = length
            if length > maxl:
                maxl = length

            avg += length
            cnt += 1

            # # print(len(words),words)
            if cutted_sentence == "":
                continue

            with open(des_file_name, 'a+') as f_out:
                f_out.writelines(cutted_sentence + separator)

            if cnt % 100 == 0:
                print(cnt)

    print(minl, maxl, avg / cnt, cnt)


jieba.add_word('对吧', tag='z')


def test():
    # jieba.del_word('信名')
    # jieba.del_word('叫屈')
    # jieba.del_word('照发')
    # jieba.del_word('没事')
    stopwords = get_stopwords()

    # cut_documents('../data/dx_xls/all1.dat', '../data/dx_xls/all1_cut.dat', stopwords)
    # cut_documents('../data/dx_xls/all1.dat', '../data/dx_xls/all1_cut_no_tag.dat', stopwords, with_tag=False)
    cut_documents('../data/dx_result/all1.dat', '../data/dx_result/all1_cut.dat', stopwords)
    # print(cut_sentence('手机号码是没事的哈您微信名字叫那个没拍对吧也都了解了对吧跟买新车一样我们我我叫益鑫集团任洪,请问王先生，我是益鑫集团的销售姓李我姓高工号是108349，介绍车抵贷产品，我们是拍拍贷合作方，你明白了吧？你知道了吧？广州河源的大绿本在受理部，2000毛钱，能不能办理，不到200，办理下来，微信说好吧，那我姓高高小慢，我姓高工号是108349',
    #                    drop_threshold=0,
    #                    sw_set=stopwords))

# test()

# for word, flag in pseg.cut('办那个宝马 X5那个车哈，'):
#     print(word, flag)
