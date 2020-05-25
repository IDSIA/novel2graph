import argparse
import sys
from dealias import Dealias
from embedding import static_dynamic_embed, plot_dist, plot_traj
import numpy as np

def main(input_file, outputfile, all_names, clusters):
    dealias = Dealias(input_file, outputfile, all_names, clusters)
    dealias.read_data()
    # best result with 20 chapters for LW and 12 for HP
    file_name, model_path, alias = static_dynamic_embed(input_file, dealias.novel.cluster_repetitions, dynamic=True, chapters=12)

    data = np.load(file_name, allow_pickle=True)
    new_data = {}
    names = data['a']
    values = data['b']
    #use these indexes to see HP main characters or see ./Data/clust&Dealias/hp1/hp1_clusters.csv to select new characters id
    # indices = [19, 0, 21, 39]
    # use these indices for LW
    indices = [34, 0, 6, 28]
    new_data['a'] = [names[indices[0]], names[indices[1]], names[indices[2]], names[indices[3]]]
    new_data['b'] = [values[indices[0]], values[indices[1]], values[indices[2]], values[indices[3]]]
    plot_dist(model_path, [alias[indices[0]], alias[indices[1]], alias[indices[2]], alias[indices[3]]], names_to_plot=new_data['a'])
    plot_traj(new_data, number_of_character=4)

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