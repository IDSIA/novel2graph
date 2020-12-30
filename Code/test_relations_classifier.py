import argparse
import sys
from relations_cls import Relations_cls
import pandas as pd

#run this file with firstBook.txt secondBook.txt (e.g. hp1.txt 03_LittleWomen.txt)
def main(input_file, new_file, output_file, all_names, clusters):
    cls = Relations_cls(input_file, new_file, output_file, all_names, clusters, n_chars=2)
    cls.set_clusters(['have', 'be', 'go'])
    cls.test()

    df = pd.read_pickle('.\\..\\Data\\embedRelations\\03_LittleWomen\\03_LittleWomen_new_relations_clusters.pkl')
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file receives a book (input) '
                                                 'and discovers the main characters of the book.')
    parser.add_argument("input_filename", type=str, help="The input file name \'file.txt\' or folder name")
    try:
        filename_book = sys.argv[1].split('.')[0]
    except:
        print('Provide an input file! Or try \'-h\' option')
        exit(-1)
    parser.add_argument("new_filename", type=str, help="The new book name \'file.txt\' or folder name")
    try:
        filename_new_book = sys.argv[2].split('.')[0]
    except:
        print('Provide another input file! Or try \'-h\' option')
        exit(-1)
    parser.add_argument("-o", "--out_filename", type=str, help="Lists output in out_filename",
                        default=filename_new_book + "_out.txt")
    parser.add_argument("-n", "--names_filename", type=str, help="Lists names occurrences in names_filename",
                        default=filename_new_book + "_name_occurrences.csv")
    parser.add_argument("-c", "--clusters_filename", type=str, help="Lists clusters in clusters_filename",
                        default=filename_new_book + "_clusters.csv")
    args = parser.parse_args()
    main(args.input_filename, args.new_filename, args.out_filename, args.names_filename, args.clusters_filename)
