import os
import logging
import pandas as pd
from dealias import Dealias
from relations_extractor import extract_relations, embedding_verbs
import numpy as np


def find_means(considered_clusters):
    clusters = set(considered_clusters['Label-verb'].values)
    data = []
    for cluster in clusters:
        cluster_rows = considered_clusters.loc[considered_clusters['Label-verb'] == cluster]
        mean_value = np.mean(list(cluster_rows['Embedding'].values), axis=0)
        data.append([mean_value, cluster])

    return pd.DataFrame(columns=['Centroid', 'Label-verb'], data=data)


def compute_distances(centroids, new_embeddings):
    best_values = []
    for i, row_embedding in new_embeddings.iterrows():
        best = 10000000
        best_label = None
        embedding = row_embedding['Embedding']
        for j, row_centroid in centroids.iterrows():
            dist = np.linalg.norm(row_centroid['Centroid'] - embedding)
            if dist < best:
                best = dist
                best_label = row_centroid['Label-verb']
        reliable = best < 3.
        best_values.append([best, best_label, row_embedding['Verb'], reliable])

    return pd.DataFrame(columns=['Distance', 'Cluster', 'Verb', 'Reliable'], data=best_values)


class Relations_cls:
    def __init__(self, book_name, new_book, output_file, all_names_file, clusters_file, n_chars=2):
        self.df_folder = "..\\Data\\embedRelations\\"
        self.book_name = book_name
        self.book_no_extension = book_name.split('.')[0]
        self.new_book = new_book
        self.new_book_no_extension = new_book.split('.')[0]
        self.output_file = output_file
        self.all_names_file = all_names_file
        self.cluster_file = clusters_file
        self.dataset_name = self.df_folder + self.book_no_extension + "\\" + self.book_no_extension + "_relations" + str(
            n_chars) + ".pkl"
        if not os.path.isfile(self.dataset_name):
            logging.info('First run test_relations_clustering.py to create the dataset.')

        self.dataset = pd.read_pickle(self.dataset_name)

    def set_clusters(self, clusters):
        dfs = []
        for cluster_name in clusters:
            dfs.append(self.dataset.loc[self.dataset['Label-verb'] == cluster_name])

        self.cluster_sentences = pd.concat(dfs)

    def test(self):
        dealias = Dealias(self.new_book, self.output_file, self.all_names_file, self.cluster_file)
        dealias_df = dealias.read_data()
        k = 50
        fname = '.\\..\\Data\\embedRelations\\' + self.new_book_no_extension + '\\' + self.new_book_no_extension + '_embeddings2.pkl'
        if not os.path.isfile(fname):
            grouped_aliases = extract_relations(self.new_book, dealias_df, k=k)

            embedding_verbs(self.new_book, grouped_aliases,
                       '.\\..\\Data\\embedRelations\\' + self.new_book_no_extension + '\\' + self.new_book_no_extension + '_right_char_sentences.pkl',
                       k)

        new_book_df = pd.read_pickle(fname)

        centroids_df = find_means(self.cluster_sentences)
        distances = compute_distances(centroids_df, new_book_df)
        data_path = '.\\..\\Data\\embedRelations\\' + self.new_book_no_extension + '\\' + self.new_book_no_extension + '_new_relations_clusters.pkl'
        logging.info('Saving data to:' + data_path)
        distances.to_pickle(data_path)