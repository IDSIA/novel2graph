# -*- coding: utf-8 -*-

import plotly.express as px
import os
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

#os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'


def color_for_val(val, vmin, vmax, color='GnBu'):
    '''
    :param color: Sequential: 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    '''
    if vmin >= vmax:
        raise ValueError('vmin must be less than vmax')
    pl_colorscale = plt.get_cmap(color)

    v = (val - vmin) / (vmax - vmin)  # val is mapped to v in [0,1]
    r, g, b, a = pl_colorscale(v)
    hexcolor = mpl.colors.to_hex([r, g, b, a], keep_alpha=False)
    # for dash-cytoscale we need the hex representation:
    return hexcolor


def plot_embedding(dataframe, x, y, color, title):
    return px.scatter(dataframe, x=x, y=y, color=color, title=title,
                      hover_data={'Verb': True, 'Label-verb': True, 'Phrase': True,
                                  'PCA-x': False, 'PCA-y': False, 'tSNE-x': False, 'tSNE-y': False})


def plot_trajectories(df, x_label, y_label, names):
    data = []
    numbers = [str(val) for val in np.arange(1, len(df.iloc[0][x_label]) + 1)]
    for i, char_label in df[names].items():
        x = df.loc[i][x_label]
        y = df.loc[i][y_label]

        this_numbers = [numbers[i] for i, value in enumerate(x) if value is not None]
        x = [value for value in x if value is not None]
        y = [value for value in y if value is not None]
        data.append(go.Scatter(x=x, y=y, name=char_label,
                               mode='lines+markers+text',
                               text=this_numbers, visible="legendonly"))

    layout = go.Layout(xaxis={'title': 'First dimension'},
                       yaxis={'title': 'Second dimension'})
    return go.Figure(data=data, layout=layout)


def create_graph(characters, relations):
    g = nx.DiGraph(encoding='utf-8')

    aliases = set(characters['Alias'])
    representatives_alias = []
    for alias in aliases:
        rows = characters.loc[characters['Alias'] == alias]
        occurrences = pd.to_numeric(rows['Occurrences'])
        i_max_occurrence = occurrences.argmax()
        representatives_alias.append([alias, rows.iloc[i_max_occurrence].loc['Names'], np.sum(occurrences)])
    representatives_alias = pd.DataFrame(data=representatives_alias, columns=['Alias', 'Name', 'Occurrences'])

    max_occurrences = np.max(representatives_alias['Occurrences'])
    for i, character in representatives_alias.iterrows():
        g.add_node(character.loc['Alias'], id=character.loc['Alias'], label=character.loc['Name'],
                   key=character.loc['Alias'],
                   shape="circle",
                   fontname="Arial", fontsize=35,
                   occurrences=character.loc['Occurrences'],
                   color=color_for_val(character.loc['Occurrences'], 0, max_occurrences))

    representatives_relations = []
    for alias_a in aliases:
        for alias_b in aliases:
            rels = relations[relations['Character1'] == alias_a][relations['Character2'] == alias_b]
            occurrence = len(rels)
            if occurrence > 0:
                representatives_relations.append([alias_a, alias_b, occurrence])
    representatives_relations = pd.DataFrame(data=representatives_relations, columns=['Character1', 'Character2', 'Occurrences'])

    max_occurrences = np.max(representatives_relations['Occurrences'])
    for index, row in representatives_relations.iterrows():
        g.add_edge(row['Character1'], row['Character2'], color=color_for_val(row['Occurrences'], 0, max_occurrences))

    return g
