#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from components import plot_trajectories


class Page_2:
    def __init__(self, book):
        self.chars_embedding = pd.read_pickle('./../../Data/embedding/embeddings/' + book + '_dynamic.pkl')
        self.chars_distances = pd.read_pickle('./../../Data/embedding/embeddings/' + book + '_dynamic_distances.pkl')

        filter_chars = []
        for i, name in self.chars_embedding['Name'].items():
            embeddings = self.chars_embedding.loc[self.chars_embedding['Name'] == name, 'Embedding'].values[0]
            for embed in embeddings:
                if embed is not None:
                    filter_chars.append({'label': name, 'value': i})
                    break

        plot_traj = plot_trajectories(self.chars_embedding,
                                      x_label='Trajectories_x',
                                      y_label='Trajectories_y',
                                      names='Name')

        self.layout_2 = html.Div(children=[
            html.Div([
                html.Br(),
                dcc.Link('Go back to home', href='/'),
                html.Br(),
                dcc.Link('Go to Page 1', href='/page-1')
            ]),
            dcc.Markdown('''
                    ## Characters trajectories
                    For a first empirical validation we process one novel, namely Harry Potter and the Philosopher’s Stone (HP) by J. K. Rowling.
                    Try to plot a character trajectory clicking a node on the legend. The trajectories are easily comparable if you activate more characters.
        
                    The numbers indicate the order of the points, accordingly to the novel's plot.
                    '''
                         ),
            html.Div([
                dcc.Graph(
                    id='trajectories',
                    figure=plot_traj
                )
            ]),
            dcc.Markdown('''
                    ## Characters distances
                    Since each character has a trajectory, we constructed an interactive plot showing the distances between characters.
                    Just select a character from the dropdown list to activate the plot.
                    At this point you can activate other "distances plots" from the legend.
                    '''
                         ),
            html.Div([
                dcc.Dropdown(
                    id='filter-distances',
                    options=filter_chars
                ),
                dcc.Graph(
                    id='distances'
                )
            ])
        ])
