#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from novel import Novel
import os
import pandas as pd

# Given a text this script look for names clusters (Ron, Ronald, Ronald Winsley,..) and dealias the original text,
# replacing each name with the corresponding cluster tag (#CHARACTER0, #CHARACTER1,...).
# You can run it passing a book file's name or a folder in which there are more books.
# Data have to be in Book folder and the result is provided in Data/clust&Dealias, in this folder you'll find a file
# with a names-occurrences list, a similar file in which few repetitions are dropped, a file with clusters and
# repetitions and finally the dealiased text.
# if you provide a folder instead of a book, texts are read and the result folder will always contain only 4 files
# (not a set of file for each book!)


class Dealias:
    def __init__(self, input_file, outputfile, all_names, clusters):
        #input_file is a file (harry_potter.txt) or a folder with some txt files (hp)
        self.input_file = input_file
        self.output_file = outputfile
        self.all_names = all_names
        self.clusters = clusters

    def read_more_texts(self, book_folder, out_folder):
        text = ''
        for book in os.listdir(book_folder):
            print('Reading:', book)
            split_name = book.split('.')
            folder_name = book_folder.split('/')[-2]
            if split_name[1] == 'txt' and split_name[0] != folder_name:
                file = open(book_folder + book, 'r', encoding='utf8')
                text += file.read()
                file.close()
        self.input_file = self.input_file + ".txt"

        all_texts = open(book_folder + self.input_file, "w+", encoding='utf8')
        all_texts.write(text)
        all_texts.close()
        self.analyze_text(book_folder, out_folder)

    def analyze_text(self, book_folder, out_folder):
        filename = self.input_file.split('.')[0]
        result_book_folder = out_folder + filename + "/"
        if not os.path.exists(result_book_folder):
            os.makedirs(os.path.dirname(result_book_folder))

        novel = Novel(book_folder + self.input_file)
        novel.read()
        novel.parse_persons()
        novel.find_persons_title()
        novel.store(filename=result_book_folder + self.all_names, data=novel.persons)
        # if you do not remove single occurrences, eps behaviour will be unstable
        occurrence_limit = 2
        novel.remove_less_than(occurrences=occurrence_limit)
        novel.store(filename=result_book_folder + filename + "_names_more_than_" + str(occurrence_limit) + ".csv",
                    data=novel.persons)
        novel.cluster_aliases()
        novel.associate_simple_single_names()
        novel.associate_single_names()
        novel.store(filename=result_book_folder + self.clusters, data=novel.cluster_repetitions)
        novel.create_cluster_repetitions_df()
        novel.cluster_repetitions_df.to_pickle(result_book_folder + filename + '.pkl')
        novel.dealiases()
        novel.store(filename=result_book_folder + filename + "_dealiased.txt", data=novel.dealiased_text, type='txt')
        #Do the coreference after the dealias, because sometimes the coreference write a name just after a separation
        # and this lead to some not desired wrong situations in which name are together (e.g. "Potter,Hermione")
        novel.coreference()
        novel.store(filename=result_book_folder + self.output_file, data=novel.dealiased_text, type='txt')
        self.novel = novel
        return novel.cluster_repetitions_df

    def read_data(self):
        book = self.input_file.split('.')[0]
        if os.path.isfile('./../Data/clust&Dealias/' + book + '/' + book + '.pkl'):
            return pd.read_pickle('./../Data/clust&Dealias/' + book + '/' + book + '.pkl')
        else:
            split_name = self.input_file.split('.')
            if len(split_name) == 1:
                # Read more file in a folder
                book_folder = "./../Books/" + split_name[0] + "/"
                out_folder = "./../Data/clust&Dealias/"
                if not os.path.exists("./../Data/clust&Dealias/"):
                    os.makedirs(os.path.dirname("./../Data/clust&Dealias/"))
                method = self.read_more_texts
            else:
                book_folder = "./../Books/"
                out_folder = "./../Data/clust&Dealias/"
                method = self.analyze_text

            if not os.path.exists(out_folder):
                os.makedirs(os.path.dirname(out_folder))

            return method(book_folder, out_folder)