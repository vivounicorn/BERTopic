from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from umap import UMAP

from utils.logger import MyLogger
import numpy as np

import pandas as pd

logger = MyLogger("WARNING")


def fetch_corpus(src_file_name: str = ''):
    documents = []
    with open(src_file_name, 'r') as f:
        while True:
            can = f.readline()
            if not can:
                break
            if '担保费' in can or '录音' in can or '管理费' in can or '回访' in can:
                continue

            documents.append(can)

    return documents


class CCTopic:
    def __init__(self,
                 embedding_model_name: str = 'DMetaSoul/sbert-chinese-general-v2-distill',
                 backend: str = 'sentence_transformers',
                 min_topic_size=15,
                 threshold: float = 0.015
                 ):
        # 'paraphrase-distilroberta-base-v1'
        # 'DMetaSoul/sbert-chinese-general-v2-distill'
        if backend == 'sentence_transformers':
            self.embedding_model = SentenceTransformer(embedding_model_name)
        elif backend == 'sentence_transformers':
            self.tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
            self.embedding_model = BertModel.from_pretrained(embedding_model_name)

        self.min_topic_size = min_topic_size
        self.topic_threshold = threshold
        self.documents = None
        self.topic_model = None
        self.topic_probs = None
        self.topics = None

    def fit_model(self, doc_path='data/dx_txt/all1_cut.dat', save_path='models/distill.dat'):
        self.documents = fetch_corpus(doc_path)
        stable_umap = UMAP(n_neighbors=15,
                           n_components=5,
                           min_dist=0.0,
                           metric='cosine',
                           random_state=2022)

        self.topic_model = BERTopic(language="chinese (simplified)",
                                    min_topic_size=self.min_topic_size,
                                    umap_model = stable_umap,
                                    calculate_probabilities=True,
                                    nr_topics='auto',
                                    verbose=True)

        s_embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)

        logger.info('Begin training...')
        self.topics, self.topic_probs = self.topic_model.fit_transform(self.documents, s_embeddings)
        logger.info('End training.')
        logger.info(self.topic_model.get_topic_info())

        # 生成主题对应的标题文档
        topics_docs = pd.DataFrame({'topic': self.topics, 'doc': self.documents})
        topics_docs.to_excel('topic.xlsx')

        self.topic_model.save(save_path)

    def visualize(self, n):
        bar_chart = self.topic_model.visualize_barchart(top_n_topics=300)
        bar_chart.write_html('figures/bar_chart.html')
        logger.info(self.topic_model.get_topic(10))

        first_new_topic_probs = self.topic_model.visualize_distribution(self.topic_probs[n], min_probability=0)
        logger.info('document content ' + self.documents[n])
        first_new_topic_probs.write_html('figures/first_new_topic_probs.html')

        visualize_topics1 = self.topic_model.visualize_topics()
        # 可视化结果保存至html中，可以动态显示信息
        visualize_topics1.write_html('figures/distance.html')

        visualize_hierarchy = self.topic_model.visualize_hierarchy()
        visualize_hierarchy.write_html('figures/hierarchy.html')

    def inference(self, docs, topn=5, model_path='models/distill.dat'):
        if self.topic_model is None:
            self.topic_model = BERTopic.load(model_path)

        can_embeddings = self.embedding_model.encode(docs)
        topics, probs = self.topic_model.transform(docs, can_embeddings)

        self.topic_threshold = 0
        selected_topics = np.argwhere(probs[0] >= self.topic_threshold).flatten()

        if topn >= len(selected_topics) > 0:
            cut_probs = probs[0][selected_topics]
        else:
            selected_topics = np.argpartition(probs[0], -topn, axis=None)[-topn:]
            cut_probs = probs[0][selected_topics]

        selected_topics = selected_topics[np.argsort(-cut_probs)]

        return topics, [(self.topic_model.topic_names[t], probs[0][t]) for t in selected_topics]
