import utils.cut_and_rm_sw
from cctopic import CCTopic

if __name__ == '__main__':
    from nerpy import NERModel

    model = NERModel("bert", "shibing624/bert4ner-base-chinese")

    sentences = [
        '喂你好，请问是石超石先生对吗。',
        '哎，你好，是这样的，我这边是携程合作机构一星集团的工作人员。',
        '有看到前两天您在携程这边有提交了一个车主用钱的申请。',
        '您本人提交的吗？',
        '哎行方便您办理，了解一下车辆情况，您那个车是全款车还是按揭车。',
        '八年裸车价多少钱呢？',
        '什么品牌的？',
        '嗯，有没有。',
        '他是这个2014年五月份之前的吗，还是五月份以后的上的牌照。',
        '哎。',
        '那就是说估计还能办，只要不超半年都可以的呀。',
        '嗯。',
        '嗯。',
        '嗯。',
        '年的话差不多30万吧，应该。',
        '三年您这进口的 宝马 X1X1是吧。',
        '估计30多万。',
        '行，那我跟您说一下啊，拟人车牌都是在郑州本地对吗？',
        '我说你人车是不是都包括这个牌照，是不是都这样子？',
        '哦，没关系。',
        '嗯，没关系啊，不管说您是本地牌照还是外地牌照都能给您办。',
        '呃，费用呢也比较便宜，1万块钱用一个月是78块钱。',
        '就一天1万块钱用一天是两块六毛钱，因为是银行直接放款的，所以说。',
        '办理过程当中呢，也没有充任何手续费服务费的。',
        '申请多少也能到账多少的。',
        '办理流程就比较简单了，就对，就加一下您微信，先帮您评个车，看你车子能做多少钱的额度，你看一下方案合适不合适，如果本地牌照就比较快。',
        '当天办很当天钱能到账，你如果是外地牌照的话，差不多是两到三个工作日。',
        '都能办，嗯嗯，对。',
        '手机号是您微信号吗。',
        '我加一下您微信，然后的话根据您那个车辆。',
        '对吧，您看用多少钱，您觉得哪个合适呢。',
        '对吧，也可以。',
        '发个方案给您看一下。',
        '我加您一下，您一会儿微信通过一下好吧。',
        '嗯行没问题，那您注意开车注意安全好吧。',
        '哎，好嘞，拜拜啊。',
        '嗯，好。'
    ]
    predictions, raw_outputs, entities = model.predict(sentences)
    print(entities)
    exit(0)


    # sentence-transformers/all-MiniLM-L6-v2
    # sentence-transformers/paraphrase-MiniLM-L6-v2
    #G shibing624/text2vec-base-chinese
    # uer/sbert-base-chinese-nli
    #G hfl/chinese-macbert-base
    topic = CCTopic(embedding_model_name='shibing624/text2vec-base-chinese', nr_topics='auto')
    # topic.fit_model(doc_path='/home/dell/PycharmProjects/BERTopic/data/dx_result/all1_merge_filter_cut.datb')
    # topic.visualize(10)

    docs = [
        '喂你好，请问是石超石先生对吗。',
        '哎，你好，是这样的，我这边是携程合作机构一星集团的工作人员。',
        '有看到前两天您在携程这边有提交了一个车主用钱的申请。',
        '您本人提交的吗？',
        '哎行方便您办理，了解一下车辆情况，您那个车是全款车还是按揭车。',
        '八年裸车价多少钱呢？',
        '什么品牌的？',
        '嗯，有没有。',
        '他是这个2014年五月份之前的吗，还是五月份以后的上的牌照。',
        '哎。',
        '那就是说估计还能办，只要不超半年都可以的呀。',
        '嗯。',
        '嗯。',
        '嗯。',
        '年的话差不多30万吧，应该。',
        '三年您这进口的宝马X1X1是吧。',
        '估计30多万。',
        '行，那我跟您说一下啊，拟人车牌都是在郑州本地对吗？',
        '我说你人车是不是都包括这个牌照，是不是都这样子？',
        '哦，没关系。',
        '嗯，没关系啊，不管说您是本地牌照还是外地牌照都能给您办。',
        '呃，费用呢也比较便宜，1万块钱用一个月是78块钱。',
        '就一天1万块钱用一天是两块六毛钱，因为是银行直接放款的，所以说。',
        '办理过程当中呢，也没有充任何手续费服务费的。',
        '申请多少也能到账多少的。',
        '办理流程就比较简单了，就对，就加一下您微信，先帮您评个车，看你车子能做多少钱的额度，你看一下方案合适不合适，如果本地牌照就比较快。',
        '当天办很当天钱能到账，你如果是外地牌照的话，差不多是两到三个工作日。',
        '都能办，嗯嗯，对。',
        '手机号是您微信号吗。',
        '我加一下您微信，然后的话根据您那个车辆。',
        '对吧，您看用多少钱，您觉得哪个合适呢。',
        '对吧，也可以。',
        '发个方案给您看一下。',
        '我加您一下，您一会儿微信通过一下好吧。',
        '嗯行没问题，那您注意开车注意安全好吧。',
        '哎，好嘞，拜拜啊。',
        '嗯，好。'
    ]

    stopwords = utils.cut_and_rm_sw.get_stopwords('/home/dell/PycharmProjects/BERTopic/dicts/')

    # cutted_docs = [utils.cut_and_rm_sw.do_filter(utils.cut_and_rm_sw.cut_sentence(" ".join(docs), sw_set=stopwords), drop_threshold=0)]
    cutted_docs = [utils.cut_and_rm_sw.do_filter(utils.cut_and_rm_sw.cut_sentence(doc, sw_set=stopwords), drop_threshold=0) for doc in docs]

    # topic.topic_threshold = 0
    for doc, cnt in cutted_docs:
        docs = []
        if cnt != 0:
            docs.append(doc)
            print('--begin--')
            print(docs)
            print(topic.inference(docs))
            print('--end--')

    print(topic.topic_model.get_topic(0))
    print(topic.topic_model.get_topic(17))
    print(topic.topic_model.get_topic(93))
    print(topic.topic_model.get_topic(19))
    print(topic.topic_model.get_topic(128))
