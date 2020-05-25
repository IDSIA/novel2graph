from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import re
import torch
import scipy
from sklearn.cluster import DBSCAN
import os

class Bert_cluster:
    def __init__(self, sentences, asymmetric_sentences, result_folder, book, aliases):
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)
        self.result_folder = result_folder
        if not os.path.isdir(result_folder + book):
            os.makedirs(result_folder + book)
        self.book = book

        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentences = sentences
        self.asymmetric_sentences = asymmetric_sentences
        self.aliases = aliases

    def remove_char_from_sentences(self):
        # remove character identifier from sentences
        only_relations_sentences = []
        removed_chars = []
        asymmetric_relations_markers = []
        for sentence in self.sentences:
            if sentence in self.asymmetric_sentences:
                asymmetric_relations_markers.append(True)
            else:
                asymmetric_relations_markers.append(False)

            new_sentence = ""
            chars = []
            for word in sentence.split(' '):
                if 'CCHARACTER' in word:
                    chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                else:
                    new_sentence += word + ' '
            removed_chars.append(chars)
            if len(chars) == 1:
                print(sentence)
            only_relations_sentences.append(new_sentence)

        self.removed_chars = removed_chars
        self.only_relations_sentences = only_relations_sentences
        self.asymmetric_relations_markers = asymmetric_relations_markers

    def embedding(self):
        return self.model.encode(self.only_relations_sentences)

    def dbscan(self, embedding):
        similarities = np.empty((len(embedding), len(embedding)))
        embedding = [vector / np.linalg.norm(vector) for vector in embedding]

        for i, sentence1 in enumerate(embedding):
            for j, sentence2 in enumerate(embedding):
                similarities[i][j] = scipy.spatial.distance.cosine(sentence1, sentence2)

        db = DBSCAN(metric='precomputed', min_samples=1, algorithm='brute', eps=0.27).fit(similarities)
        return db.labels_

    def kmeans(self, sentence_embeddings, k):
        km = KMeans(n_clusters=k)
        # normalize vectors before the kmean
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in sentence_embeddings]

        km.fit(sentence_embeddings)
        return np.array(km.labels_.tolist())

    def generate_triplets(self, clusters):
        triplets = []
        for i, pair in enumerate(self.removed_chars):
            if len(pair) == 1:
                triplets.append()
            triplet = np.array([pair[0], clusters[i], pair[1]])
            triplets.append(triplet)
        self.triplets = triplets

    def generate_reports(self, clusters):
        file = open(self.result_folder + self.book + '/' + self.book + '_report.txt', "w+")
        data = {}
        for i in range(0, len(Counter(clusters).keys())):
            involved_triplet = np.where(clusters == i)
            file.write('----Relation ' + str(i) + '---\n')
            for index in involved_triplet[0]:
                current_triplet = self.triplets[int(index)]
                file.write(self.aliases[current_triplet[0]][0] + '\t' + str(current_triplet[1]) + '\t' +
                           self.aliases[current_triplet[2]][
                               0] + '\t' + str(self.asymmetric_relations_markers[index]) + '\t' + self.sentences[
                               index] + '\n')

                key = (current_triplet[0], current_triplet[2])
                if key not in data:
                    data[key] = []
                data[key].append(str(current_triplet[1]))
        file.close()
        self.chars_relations = data

    def silhouette(self, embeddings, max_iter=10):
        silh = []
        j = 2

        embeddings = [vector / np.linalg.norm(vector) for vector in embeddings]
        while j <= max_iter:
            km = KMeans(n_clusters=j, random_state=0, n_init=1, max_iter=1)
            km.fit(embeddings)
            y_kmean = km.predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, y_kmean)
            silh.append(silhouette_avg)
            j = j + 1

        # the K-value corresponding to maximum silhoutte score is selected
        m = max(silh)
        max_indices = [i for i, j in enumerate(silh) if j == m]
        maximums = max_indices[0] + 2
        return silh, maximums
