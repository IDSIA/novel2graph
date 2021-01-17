import argparse
import sys
from dealias import Dealias
from relations_extractor import extract_relations, embed_and_cluster
import pandas as pd


def main(input_file, outputfile, all_names, clusters):
    dealias = Dealias(input_file, outputfile, all_names, clusters)
    dealias_df = dealias.read_data()
    k = 50
    grouped_aliases = extract_relations(input_file, dealias_df, k=k)
    embed_and_cluster(input_file, grouped_aliases, './../Data/embedRelations/hp1/hp1_few_char_sentences.pkl', k)
    embedding_1char_df = pd.read_pickle('.\\..\\Data\\embedRelations\\hp1\\hp1_embeddings1.pkl')
    relations_1char_df = pd.read_pickle('.\\..\\Data\\embedRelations\\hp1\\hp1_relations1.pkl')

    embed_and_cluster(input_file, grouped_aliases, './../Data/embedRelations/hp1/hp1_right_char_sentences.pkl', k)
    embedding_2chars_df = pd.read_pickle('.\\..\\Data\\embedRelations\\hp1\\hp1_embeddings2.pkl')
    relations_2chars_df = pd.read_pickle('.\\..\\Data\\embedRelations\\hp1\\hp1_relations2.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file receives a book (input) '
                                                 'and discovers the main characters of the book.')
    parser.add_argument("input_filename", type=str, help="The input file name \'file.txt\' or folder name")
    try:
        filename = sys.argv[1].split('.')[0]
    except:
        print('Provide an input file! Or try \'-h\' option')
        exit(-1)
    parser.add_argument("-o", "--out_filename", type=str, help="Lists output in out_filename",
                        default=filename + "_out.txt")
    parser.add_argument("-n", "--names_filename", type=str, help="Lists names occurrences in names_filename",
                        default=filename + "_name_occurrences.csv")
    parser.add_argument("-c", "--clusters_filename", type=str, help="Lists clusters in clusters_filename",
                        default=filename + "_clusters.csv")
    args = parser.parse_args()
    main(args.input_filename, args.out_filename, args.names_filename, args.clusters_filename)
