import torch
import torch.nn.functional as F
import umap
from sklearn.datasets import load_digits
import umap.plot
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
from sentence_transformers import SentenceTransformer
import spacy
from flair.embeddings import XLMEmbeddings
import numpy as np
import utils.cut_and_rm_sw
from cctopic import CCTopic


def test_embedding():
    sentences = ["我这边是易鑫集团的工作人员", "办理流程就比较简单了?", "裸车价多少钱呢？", "加一下您微信"]

    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    embeddings = torch.from_numpy(model.encode(sentences))

    cand = embeddings[0]
    print(sentences[0])

    for i in range(1, len(embeddings)):
        print('——', sentences[i], '相关度:', F.cosine_similarity(cand, embeddings[i], dim=0))

    from flair.data import Sentence
    from flair.embeddings import BertEmbeddings

    # init embedding
    embedding = BertEmbeddings('bert-base-chinese')

    sentences = ["我这边是易鑫集团的工作人员", "办理流程就比较简单了?", "裸车价多少钱呢？", "加一下您微信"]

    # create a sentence
    sentence = [Sentence(s) for s in sentences]
    embedding.embed(sentence[0])
    print(sentence[0].embedding)

    embedding = XLMEmbeddings(pooling_operation="mean", use_scalar_mix=True)

    sentence = Sentence(['The Hofbräuhaus is a beer hall in Munich .'])
    embedding.embed(sentence)
    print(sentence[0].embedding)

    nlp = spacy.load("zh_core_web_sm")
    doc = [nlp(s) for s in sentences]
    for i in range(1, len(doc)):
        print('——', sentences[i], '相关度:', doc[0].similarity(doc[i]))


def test_digits():
    digits = load_digits()
    f, a = plt.subplots(15, 15)
    axes = a.flatten()
    for i, item in enumerate(axes):
        item.imshow(digits.images[i], cmap='gray')

    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.05, w_pad=0.02)

    mapper = umap.UMAP(n_neighbors=15,
                       n_components=2,
                       min_dist=0.1,
                       random_state=2022).fit_transform(digits.data)
    # p = umap.plot.connectivity(mapper, show_points=True, labels=digits.target)
    # umap.plot.show(p)
    # q = umap.plot.points(mapper)
    # umap.plot.show(q)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10,
                                metric='manhattan',
                                cluster_selection_method='eom',
                                prediction_data=True)
    clusterer.fit(mapper)
    clusterer.condensed_tree_.plot(select_clusters=True)
    ids = {}
    for x in clusterer.labels_:
        if x not in ids.keys():
            ids[x] = 1

    # print(clusterer.labels_)
    color_palette = sns.color_palette('bright', len(ids))
    # print(color_palette)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]

    plt.scatter(mapper[:, 0], mapper[:, 1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()


def test_hdbscan():
    data = np.loadtxt('/home/dell/PycharmProjects/BERTopic/data/usps/c.data')
    # plt.scatter(*data.T, s=50, linewidth=0, c='b', alpha=0.25)
    clusterer = hdbscan.HDBSCAN(
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True).fit(data)

    ids = {}
    for x in clusterer.labels_:
        if x not in ids.keys():
            ids[x] = 1

    print(ids)

    color_palette = sns.color_palette('deep', 8)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.scatter(*data.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    clusterer.condensed_tree_.plot(select_clusters=True,
                                   selection_palette=sns.color_palette('deep', 8))
    plt.show()


def test_bt():
    topic = CCTopic(embedding_model_name='shibing624/text2vec-base-chinese', nr_topics='auto')
    topic.fit_model(doc_path='/home/dell/PycharmProjects/BERTopic/data/zhidao/20130820_filter_cut.dat')
    topic.visualize(10)


# test_embedding()
# test_digits()
# test_hdbscan()

test_bt()