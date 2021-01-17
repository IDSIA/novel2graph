#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from style_gui import default_stylesheet, text_style
import os.path
from components import plot_embedding, create_graph
import networkx as nx
import logging
import pandas as pd


class Page_1:
    def __init__(self, book):
        self.file_name1 = "..\\..\\Data\\clust&Dealias\\" + book + "\\" + book + "_characters.pkl"
        self.file_name2 = "..\\..\\Data\\embedRelations\\" + book + "\\" + book + "_relations2.pkl"
        self.file_name3 = "..\\..\\Data\\embedRelations\\" + book + "\\" + book + "_relations1.pkl"
        if not os.path.isfile(self.file_name1) or not os.path.isfile(self.file_name2) or not os.path.isfile(
                self.file_name3):
            logging.info('First run test_relations_clustering.py to create the dataset.')
            exit(-1)

        self.characters = pd.read_pickle(self.file_name1)
        two_chars_relations = pd.read_pickle(self.file_name2)
        self.two_chars_relations = two_chars_relations.sort_values('Label-verb')
        one_char_relations = pd.read_pickle(self.file_name3)
        self.one_char_relations = one_char_relations.sort_values('Label-verb')

        relations_attributes = ['Character1', 'Verb', 'Label-verb', 'Character2', 'Phrase']
        nodes_attributes = ['Alias', 'Names', 'Occurrences']

        two_chars_relations = two_chars_relations.round({'PCA-x': 2, 'PCA-y': 2})
        one_char_relations = one_char_relations.round({'PCA-x': 2, 'PCA-y': 2})

        embedding_pca = plot_embedding(two_chars_relations, x='PCA-x', y='PCA-y', color='Label-verb',
                                       title='2-D PCA projection')
        embedding_tsne = plot_embedding(two_chars_relations, x='tSNE-x', y='tSNE-y', color='Label-verb',
                                        title='2-D t-SNE projection')
        graph = create_graph(self.characters, two_chars_relations)
        graph_data = nx.readwrite.json_graph.cytoscape_data(graph)

        self.layout_1 = html.Div(children=[
            html.Div([
                html.Br(),
                dcc.Link('Go back to home', href='/'),
                html.Br(),
                dcc.Link('Go to Page 2', href='/page-2')
            ]),

            html.Div([
                dcc.Markdown('''
                    ## Relation graph
                    For a first empirical validation of the pipeline described in the previous section we process one novel,
                    namely Harry Potter and the Philosopher’s Stone (HP) by J. K. Rowling.
                    In this graph each node represent a character, and an edge between characters means that there is at least one
                    relation involving the two characters. The stronger the color, the more important the character or the relation.
        
                    Note that in the following section you can select different graphical representation, to see the same data structures.
                    Clicking a node you will see its details (repetitions, aliases,...) and clicking a relation you will see a list
                    of similar relations with some details. You can also sorting and filter each table using its header.
                    '''
                             ),
                html.Div([
                    dcc.Dropdown(
                        id='dropdown-update-layout',
                        value='grid',
                        clearable=False,
                        options=[
                            {'label': name.capitalize(), 'value': name}
                            for name in ['grid', 'random', 'circle', 'cose', 'concentric', 'breadthfirst']
                        ]
                    )
                ], style=text_style),
                html.Div([
                    cyto.Cytoscape(
                        id='cytoscape-update-layout',
                        layout={'name': 'grid'},
                        style={'width': '100%', 'height': '450px'},
                        stylesheet=default_stylesheet,
                        elements=graph_data['elements']
                    )
                ])
            ], className="row"),
            html.Div([
                html.Div(id='cytoscape-tapNodeData-output'),
                dash_table.DataTable(
                    id='nodes-table',
                    columns=[{"name": i, "id": i} for i in nodes_attributes],
                    sort_action='native',
                    filter_action='native',
                    style_cell_conditional=[
                        {
                            'if': {'column_id': c},
                            'textAlign': 'left'
                        } for c in nodes_attributes
                    ],
                    style_data={
                        'whiteSpace': 'normal',
                    },
                    css=[{
                        'selector': '.dash-spreadsheet td div',
                        'rule': '''
                                            line-height: 15px;
                                            max-height: 30px; min-height: 30px; height: 30px;
                                            display: block;
                                            overflow-y: hidden;
                                        '''
                    }],
                    style_cell={'textAlign': 'left'},
                    style_as_list_view=True,
                )

            ], className="row"),
            html.Div([
                html.Div([
                    html.H4('Non-relational clustering'),
                    html.Div(id='cytoscape-single-relations'),
                    dash_table.DataTable(
                        id='single-relation-table',
                        columns=[{"name": i, "id": i} for i in relations_attributes],
                        sort_action='native',
                        filter_action='native',
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in relations_attributes
                        ],
                        style_data={
                            'whiteSpace': 'normal',
                        },
                        css=[{
                            'selector': '.dash-spreadsheet td div',
                            'rule': '''
                                            line-height: 15px;
                                            max-height: 30px; min-height: 30px; height: 30px;
                                            display: block;
                                            overflow-y: hidden;
                                        '''
                        }],
                        style_cell={'textAlign': 'left'},
                        style_as_list_view=True,
                    )
                ], className="six columns"),

                html.Div([
                    html.H4('Relational clustering'),
                    html.Div(id='cytoscape-tapRelationData-output'),
                    dash_table.DataTable(
                        id='relations-table',
                        columns=[{"name": i, "id": i} for i in relations_attributes],
                        sort_action='native',
                        filter_action='native',
                        style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in relations_attributes
                        ],
                        style_data={
                            'whiteSpace': 'normal',
                        },
                        css=[{
                            'selector': '.dash-spreadsheet td div',
                            'rule': '''
                                        line-height: 15px;
                                        max-height: 30px; min-height: 30px; height: 30px;
                                        display: block;
                                        overflow-y: hidden;
                                    '''
                        }],
                        style_cell={'textAlign': 'left'},
                        style_as_list_view=True,
                    )
                ], className="six columns")
            ], className="row"),
            html.Div([
                dcc.Markdown('''
                ## Verbs embedding
                In order to identify similar relations between two characters, we analyse the first verb in each phrase containing exactly two entities. 
                Embedding each verb, we obtain a list of N-Dimensional vectors, which is plotted below using a 2-D PCA projection and a 2-D t-SNE projection.
                Finally, each N-Dimensional vector is also clustered with similar vectors, using the K-Mean algorithm.
                So in conclusion we obtain a 2-D representation of verbs, which are colored accordingly with their cluster affiliation.
        
                Using the legend you can show/hide each cluster, or simply double click on a point to see others cluster components.
                '''),
                html.Div([
                    dcc.Graph(
                        id='embedding_pca',
                        figure=embedding_pca
                    )
                ]),
                html.Div([
                    dcc.Graph(
                        id='embedding_tsne',
                        figure=embedding_tsne
                    )
                ])
            ], className="row")
        ])
