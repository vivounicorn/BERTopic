# coding=utf-8
import os

import jieba
import jieba.analyse
import jieba.posseg as pseg
from pycorrector.macbert.macbert_corrector import MacBertCorrector
import re

jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/user_words.txt')
jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/car_brand.txt')
jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/financial_institution.txt')
jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/place_name.txt')
jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/common_name.txt')
jieba.load_userdict('/home/dell/PycharmProjects/BERTopic/dicts/financial_vocabulary.txt')

correct_mistakes = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct


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


def cut_sentence(can: str = '', sw_set: set = None):
    words = pseg.cut(can)

    cutted_words = []
    for word, flag in words:
        if word not in sw_set:
            cutted_words.append(word + '/' + flag)

    return " ".join(cutted_words)


def cut_documents(src_file_name: str, des_file_name: str, sw_set: set, drop_threshold=1, separator='\r'):
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

            try:
                can = correct_mistakes(can)[0]
                cutted_sentence = cut_sentence(can, sw_set=sw_set)
            except:
                print('error.')
                continue

            cnt += 1

            if cutted_sentence == "":
                continue

            with open(des_file_name, 'a+') as f_out:
                f_out.writelines(cutted_sentence + separator)

            if cnt % 100 == 0:
                print(cnt)

    print(cnt)


def do_filter(line: str, drop_threshold=1):
    cutted_words = []
    words_list = line.split(" ")
    for word in words_list:
        wf_pair = word.split('/')
        if len(wf_pair) != 2:
            continue

        word = wf_pair[0]
        flag = wf_pair[1]

        # 	Ag	|	形语素	|	形容词性语素。 形容词代码为 a ,语素代码 g 前面置以 A。
        # 	a	|	形容词	|	取英语形容词 adjective 的第 1 个字母。
        # 	ad	|	副形词	|	直接作状语的形容词。 形容词代码 a 和副词代码 d 并在一起。
        # 	an	|	名形词	|	具有名词功能的形容词。 形容词代码 a 和名词代码 n 并在一起。
        # 	b	|	区别词	|	取汉字“别”的声母。
        # 	c	|	连词	    |	取英语连词 conjunction 的第 1 个字母。
        # 	Dg	|	副语素	|	副词性语素。 副词代码为 d ,语素代码 g 前面置以 D。
        # 	d	|	副词	    |	取 adverb 的第 2 个字母 ,因其第 1 个字母已用于形容词。
        # 	e	|	叹词	    |	取英语叹词 exclamation 的第 1 个字母。
        # 	f	|	方位词	|	取汉字“方” 的声母。
        # 	g	|	语素	    |	绝大多数语素都能作为合成词的“词根”,取汉字“根”的声母。 由于实际标注时 ,一定标注其子类 ,所以从来没有用到过 g。
        # 	h	|	前接成分	|	取英语 head 的第 1 个字母。
        # 	i	|	成语	    |	取英语成语 idiom 的第 1 个字母。
        # 	j	|	简称略语	|	取汉字“简”的声母。
        # 	k	|	后接成分	|
        # 	l	|	习用语	|	习用语尚未成为成语 ,有点“临时性”,取“临”的声母。
        # 	m	|	数词	    |	取英语 numeral 的第 3 个字母 ,n ,u 已有他用。
        # 	Ng	|	名语素	|	名词性语素。 名词代码为 n ,语素代码 g 前面置以 N。
        # 	n	|	名词	    |	取英语名词 noun 的第 1 个字母。
        # 	nr	|	人名	    |	名词代码 n 和“人(ren) ”的声母并在一起。
        # 	ns	|	地名	    |	名词代码 n 和处所词代码 s 并在一起。
        # 	nt	|	机构团体	|	“团”的声母为 t，名词代码 n 和 t 并在一起。
        # 	nx	|	非汉字串	|
        # 	nz	|	其他专名	|	“专”的声母的第 1 个字母为 z，名词代码 n 和 z 并在一起。
        # 	o	|	拟声词	|	取英语拟声词 onomatopoeia 的第 1 个字母。
        # 	p	|	介词	    |	取英语介词 prepositional 的第 1 个字母。
        # 	q	|	量词	    |	取英语 quantity 的第 1 个字母。
        # 	r	|	代词	    |	取英语代词 pronoun 的第 2 个字母,因 p 已用于介词。
        # 	s	|	处所词	|	取英语 space 的第 1 个字母。
        # 	Tg	|	时语素	|	时间词性语素。时间词代码为 t,在语素的代码 g 前面置以 T。
        # 	t	|	时间词	|	取英语 time 的第 1 个字母。
        # 	u	|	助词	    |	取英语助词 auxiliary 的第 2 个字母,因 a 已用于形容词。
        # 	Vg	|	动语素	|	动词性语素。动词代码为 v。在语素的代码 g 前面置以 V。
        # 	v	|	动词	    |	取英语动词 verb 的第一个字母。
        # 	vd	|	副动词	|	直接作状语的动词。动词和副词的代码并在一起。
        # 	vn	|	名动词	|	指具有名词功能的动词。动词和名词的代码并在一起。
        # 	w	|	标点符号	|
        # 	x	|	非语素字	|	非语素字只是一个符号，字母 x 通常用于代表未知数、符号。
        # 	y	|	语气词	|	取汉字“语”的声母。
        # 	z	|	状态词	|	取汉字“状”的声母的前一个字母。
        # flags = ['n', 'a', 'v', 'ad', 'vd', 'an', 't', 'vn']
        flags = ['n', 'a', 'an', 'vn', 'v']

        if (len(flags) == 0 or flag in flags) and (not re.match('^[0-9|.]*$', word)) and (len(word) > 1):
            cutted_words.append(word)

    if len(cutted_words) < drop_threshold:
        return "", 0

    return " ".join(cutted_words), len(cutted_words)


def cutted_filter(infile_path, outfile_path):
    if os.path.exists(outfile_path):
        os.remove(outfile_path)
    cnt = 0
    with open(infile_path) as f:
        for line in f.readlines():
            w, _ = do_filter(line)
            if w == "":
                continue

            with open(outfile_path, 'a+') as f_out:
                f_out.writelines(w + '\n')

            cnt += 1
            if cnt % 100 == 0:
                print(cnt)


jieba.add_word('对吧', tag='z')


def cut_batch(cans):
    stopwords = set()#get_stopwords()
    for can in cans:
        res = cut_sentence(can, sw_set=stopwords)
        print(res)


def run_cutting(infile_path, outfile_path):
    stopwords = get_stopwords()
    cut_documents(infile_path, outfile_path, stopwords)


if __name__ == '__main__':
    # org_in_file = '../data/dx_result/all1.dat'
    # org_out_file = '../data/dx_result/all1_with_flag_cut.dat'
    # cutted_file = '../data/dx_result/all1_filter_cut.dat'
    # org_in_file = '../data/dx_result/all1_merge.dat'
    # org_out_file = '../data/dx_result/all1_merge_with_flag_cut.dat'
    # cutted_file = '../data/dx_result/all1_merge_filter_cut.dat'
    org_in_file = '../data/zhidao/20130820'
    org_out_file = '../data/zhidao/20130820_with_flag_cut.dat'
    cutted_file = '../data/zhidao/20130820_filter_cut.dat'

    run_cutting(org_in_file, org_out_file)
    cutted_filter(org_out_file, cutted_file)
    # print(correct_mistakes(";幺五年的是吧。;幺五年哪个品牌呢？;您拨打的用户已启用天翼通讯助理漏话提醒服务。;我们将尽快通知对方，感谢您的来电。"))
    # test(";喂你好，我这边是鄂尔多斯火车站益兴集团的回访人员，请问是何先生对吧。;哎，打扰一下，做个简单的回访，请问您收到全部放款了吗。;跟您核对一下是7万元金额对吗。;嗯，那您在办业务的过程中有没有支付过一些额外的费用啊？;除了有个300元的费用之外，有人向您收过服务费、手续费之类的吗？;好的，那工作人员有让您关注微信公众号吗。;最后问一下，您对本次业务的服务是否满意？;好的，感谢您的接听啊，再见。")

    docs = []
    cut_batch(docs)

