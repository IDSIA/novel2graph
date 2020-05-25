import argparse
import sys
from dealias import Dealias
from relations_extractor import extract_relations
from bart_summarization_pipeline import Triplets


#TODO find a new version of hp3
def main(input_file, outputfile, all_names, clusters):
    dealias = Dealias(input_file, outputfile, all_names, clusters)
    dealias.read_data()

    grouped_aliases = extract_relations(input_file, dealias.novel.cluster_repetitions)
    triplets = Triplets('./../Data/embedRelations/', input_file, grouped_aliases)
    triplets.summarize()
    triplets.triplet_generate()


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