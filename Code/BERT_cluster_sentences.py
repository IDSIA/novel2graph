from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering,MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.metrics import silhouette_score
import re
import torch
import scipy
from sklearn.cluster import DBSCAN
import os
import matplotlib.pyplot as plt
import logging
from collections import Counter
import pandas as pd


def find_cluster_verb(involved_triplet, verbs):
    cluster_verbs = [verbs[index] for index in involved_triplet[0]]
    return Counter(cluster_verbs).most_common(1)[0][0]


class Bert_cluster:
    def __init__(self, result_folder, book, aliases, sentences_file=None):
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)
        self.result_folder = result_folder
        if not os.path.isdir(result_folder + book):
            os.makedirs(result_folder + book)
        self.book = book

        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aliases = aliases
        if sentences_file is not None:
            self.sentences = pd.read_pickle(sentences_file)
            sentences_str = [' '.join(sent) for sent in self.sentences['Original_token']]
            sentences_str = [sentence.replace('\n', '') for sentence in sentences_str]
            self.sentences_str = pd.Series(sentences_str)
            sentences_str = [' '.join(sent) for sent in self.sentences['Lemmatize_token']]
            sentences_str = [sentence.replace('\n', '') for sentence in sentences_str]
            self.sentences_lemmatize_str = pd.Series(sentences_str)

            self.chars_number = len(self.sentences.iloc[0]['Characters'])

    def extract_verbs(self):
        verbs_between_chars = {}
        removed_chars = {}
        new_sentences = {}
        for id, row in self.sentences.iterrows():
            try:
                sentence_str = self.sentences_str.iloc[id]
                sentence_lemmatize_str = self.sentences_lemmatize_str.iloc[id]

            except:
                print("An exception occurred while processing this sentence.")
                exit(-1)
            tokens = row['Original_token']
            tags = row['Tag']
            #list of lemmatize verbs
            verbs = []
            #the original tag for each item in verbs
            verbs_tag = []
            chars = []

            if self.chars_number == 2:
                between_names = False
                continues_verb = False
                first_verb = True

                for i, word in enumerate(tokens):
                    if 'CCHARACTER' in word:
                        chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                        between_names = not between_names
                        continue
                    if between_names:
                        if 'VB' in tags[i]:
                            if first_verb:
                                verbs.append(row['Lemmatize_token'][i])
                                verbs_tag.append(tags[i])
                                first_verb = False
                                continues_verb = True
                                continue
                            if not first_verb and continues_verb:
                                verbs.append(row['Lemmatize_token'][i])
                                verbs_tag.append(tags[i])
                            else:
                                continue
                        else:
                            continues_verb = False
            #single character in the sentence
            elif self.chars_number == 1:
                continues_verb = False
                first_verb = True
                for i, word in enumerate(tokens):
                    if 'CCHARACTER' in word:
                        chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                        continue
                    if 'VB' in tags[i]:
                        if first_verb:
                            verbs.append(row['Lemmatize_token'][i])
                            verbs_tag.append(tags[i])
                            first_verb = False
                            continues_verb = True
                            continue
                        if not first_verb and continues_verb:
                            verbs.append(row['Lemmatize_token'][i])
                            verbs_tag.append(tags[i])
                        else:
                            continue
                    else:
                        continues_verb = False
            else:
                logging.info('Are you sure about the number of characters? If so, implement this part.')
                exit(-1)

            verbs_str = ' '.join(verbs)
            if verbs_str != '':
                verb = None
                for i in range(len(verbs)-1, -1, -1):
                    if 'VB' in verbs_tag[i]:
                        verb = verbs[i]
                        break
                if verb is not None:
                    if len(chars) == self.chars_number:
                        verbs_between_chars[id] = verb
                        removed_chars[id] = chars
                        new_sentences[id] = sentence_str
                    else:
                        logging.info('Someting went wrong with the amount of characters you provide: %s, names: %s', sentence_str, chars)
                else:
                    logging.info('VB not detected in sentence: %s', sentence_str)
            else:
                logging.info('Verb not detected: %s', sentence_str)

        self.verbs = verbs_between_chars
        self.removed_chars = removed_chars
        self.sentences_dictionary = new_sentences

    def remove_char_from_sentences(self):
        # Deprecated, not working anymore
        # remove character identifier from sentences
        no_char_sentences = {}
        removed_chars = []
        for id, sentence in self.sentences_dictionary.items():
            new_sentence = ""
            chars = []
            for word in sentence.split(' '):
                if 'CCHARACTER' in word:
                    chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                    continue
                new_sentence += word + ' '
            no_char_sentences[id] = new_sentence
            removed_chars.append(chars)
        self.removed_chars = removed_chars
        self.no_char_sentences = no_char_sentences

    def embedding(self):
        embedding = self.model.encode(list(self.verbs.values()))
        id_embedding = {}
        i = 0
        for id, verb in self.verbs.items():
            id_embedding[id] = embedding[i]
            i += 1
        self.embedding = id_embedding

    def dbscan(self, metric='precomputed', min_samples=1, algorithm='brute', eps=0.27):
        embedding = list(self.embedding.values())
        similarities = np.empty((len(embedding), len(embedding)))
        embedding = [vector / np.linalg.norm(vector) for vector in embedding]

        for i, sentence1 in enumerate(embedding):
            for j, sentence2 in enumerate(embedding):
                similarities[i][j] = 1. + scipy.spatial.distance.cosine(sentence1, sentence2)

        db = DBSCAN(metric=metric, min_samples=min_samples, algorithm=algorithm, eps=eps).fit(similarities)
        self.cluster_model = db
        return db.labels_

    def affinity_propagation(self):
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        af = AffinityPropagation().fit(sentence_embeddings)

        return af.labels_

    def mean_shift(self):
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        bandwidth = estimate_bandwidth(sentence_embeddings)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(sentence_embeddings)

        return ms.labels_

    def kmeans(self, k):
        km = KMeans(n_clusters=k)
        # normalize vectors before the kmean
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]

        km.fit(sentence_embeddings)
        self.cluster_model = km
        return np.array(km.labels_.tolist())

    def generate_triplets(self, clusters):
        triplets = []
        i = 0
        print(len(clusters), len(self.removed_chars), len(self.verbs), len(self.sentences_dictionary))
        for key, value in self.removed_chars.items():
            if self.chars_number == 2:
                triplet = np.array([value[0], clusters[i], value[1]])
            elif self.chars_number == 1:
                triplet = np.array([value[0], clusters[i]])
            else:
                logging.info('Are you sure about the number of characters? If so, implement this part.')
                exit(-1)
            triplets.append(triplet)
            i += 1
        self.triplets = triplets

    def label_verbs(self, clusters):
        verbs = list(self.verbs.values())
        labels = {}
        for i in range(0, len(Counter(clusters).keys())):
            involved_triplet = np.where(clusters == i)
            if len(involved_triplet[0]) == 0:
                logging.info('----No Relation---\n')
                continue
            verb = find_cluster_verb(involved_triplet, verbs)
            labels[i] = verb
        return labels

    def generate_reports(self, clusters):
        file = open(self.result_folder + self.book + '/' + self.book + '_report_' + str(self.chars_number) + '.txt', "w+")
        data = {}
        all_cluster = {}
        sentences = list(self.sentences_dictionary.values())
        verbs = list(self.verbs.values())
        for i in range(0, len(Counter(clusters).keys())):
            involved_triplet = np.where(clusters == i)
            if len(involved_triplet[0]) == 0:
                file.write('----No Relation---\n')
                continue
            verb = find_cluster_verb(involved_triplet, verbs)
            file.write('----Relation ' + verb + '---\n')
            all_cluster[i] = []
            for index in involved_triplet[0]:
                current_triplet = self.triplets[int(index)]
                sentence = sentences[index]
                #sentence = sentence.replace('\n', '')
                if self.chars_number == 2:
                    file.write(self.aliases[current_triplet[0]][0] + '\t' + verbs[index] + '\t' +
                               self.aliases[current_triplet[2]][
                                   0] + '\t' + sentence + '\n')

                    key = (current_triplet[0], current_triplet[2])
                    if key not in data:
                        data[key] = []
                    data[key].append(str(current_triplet[1]))
                elif self.chars_number == 1:
                    file.write(self.aliases[current_triplet[0]][0] + '\t' + verbs[index] + '\t' + sentence + '\n')
                    key = (current_triplet[0])
                    if key not in data:
                        data[key] = []
                    data[key].append(str(current_triplet[1]))
                else:
                    logging.info('Are you sure about the number of characters? If so, implement this part.')
                    exit(-1)
                all_cluster[i].append(list(self.sentences_dictionary.keys())[index])
        file.close()
        self.chars_relations = data
        self.all_cluster = all_cluster

    def silhouette_kmean(self, max_iter=50):
        silh = []
        j = 2
        best_silhouette = -1
        best_k = -1
        embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        sse = {}
        while j <= max_iter:
            km = KMeans(n_clusters=j)
            km = km.fit(embeddings)
            y = km.predict(embeddings)

            sse[j] = km.inertia_

            silhouette_avg = silhouette_score(embeddings, y)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = j
            silh.append(silhouette_avg)
            j = j + 1

        # the K-value corresponding to maximum silhoutte score is selected
        m = max(silh)
        max_indices = [i for i, j in enumerate(silh) if j == m]
        maximums = max_indices[0] + 2

        fig, axs = plt.subplots(2)
        fig.suptitle('Best Kmean evalution')
        axs[0].plot(list(sse.keys()), list(sse.values()))
        axs[0].set(ylabel='SSE')
        axs[1].plot(list(sse.keys()), silh)
        axs[1].set(ylabel='Silhouette')
        plt.xlabel("Number of cluster")
        plt.show()

        return silh, maximums

    def silhouette_dbscan(self, max_min_sample=100, max_eps=5):
        silh = []
        epsi = []
        clusters = []
        eps = 0.2
        embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        best_silh = -1

        while eps <= max_eps:
            y = self.dbscan(min_samples=1, eps=eps)
            if max(y) < 2:
                eps = eps + 0.2
                continue
            silhouette_avg = silhouette_score(embeddings, y)
            silh.append(silhouette_avg)
            epsi.append(eps)
            clusters.append(max(y))
            if silhouette_avg > best_silh:
                best_silh = silhouette_avg

            eps = eps + 0.2

        # the K-value corresponding to maximum silhoutte score is selected
        #m = max(silh)
        #max_indices = [i for i, j in enumerate(silh) if j == m]
        #maximums = max_indices[0] + 2


        #fig.suptitle('Best Kmean evalution')
        plt.plot(epsi, silh)
        plt.ylabel("Avg. silhouette")
        plt.xlabel("Evolution of epsilon")
        plt.show()

        #return silh, maximums
