#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
from page_1 import Page_1
from page_2 import Page_2
import logging
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import argparse
import sys

logging.getLogger().setLevel(logging.INFO)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
index_page = html.Div([
    html.Div([
        html.Br(),
        dcc.Link('Go to Page 1', href='/page-1'),
        html.Br(),
        dcc.Link('Go to Page 2', href='/page-2')
    ]),
    html.Div([
        dcc.Markdown('''
            # Novel2graph
            #### A Web Application for novel analysis, by Simone Mellace.
            When coping with literary texts such as novels or short stories, the extraction of structured information in the form of a 
            knowledge graph might be hindered by the huge number of possible relations between the entities corresponding to 
            the characters in the novel and the consequent hurdles in gathering supervised information about them. This Web 
            Application presents some methods depicted and used in [Temporal Embeddings and 
            Transformer Models for Narrative Text Understanding](https://arxiv.org/abs/2003.08811) and [Relation Clustering in 
            Narrative Knowledge Graphs](https://arxiv.org/abs/2011.13647).''', className="row"
        ),
        dcc.Markdown('''
            Results of Temporal Embeddings and Transformer Models for Narrative Text Understanding are shown at [Page 1](http://127.0.0.1:8050/page-1), while Relation Clustering in 
            Narrative Knowledge Graphs is exposed at [Page 2](http://127.0.0.1:8050/page-2).'''
        )
    ])
]),
app.layout = html.Div([
    html.Div(id='page-content'),
    dcc.Location(id='url', refresh=False)])





@app.callback(Output('cytoscape-tapRelationData-output', 'children'),
              [Input('cytoscape-update-layout', 'tapEdgeData')])
def displayTapRelationData(data):
    if data:
        from_char = characters[characters['Alias'] == data['source']]['Names'].values[0]
        to_char = characters[characters['Alias'] == data['target']]['Names'].values[0]
        return "You are seeing the phrases involving a relation from " + from_char + ' to ' + to_char + '.'
    else:
        return "Click a relation to see its cluster phrases."


@app.callback(Output('cytoscape-single-relations', 'children'),
              [Input('cytoscape-update-layout', 'tapNodeData')])
def displaySingleRelationData(data):
    if data:
        return "You are seeing the cluster of phrases involving only character: " + data['label'] + '.'
    else:
        return "Click a node to see clusters with phrases including only this character."


@app.callback(Output('cytoscape-tapNodeData-output', 'children'),
              [Input('cytoscape-update-layout', 'tapNodeData')])
def displayTapNodeData(data):
    if data:
        return "Here below the aliases of the character: " + data['label'] + '.'
    else:
        return "Please, click a node to see its aliases."


@app.callback(Output('cytoscape-update-layout', 'layout'),
              [Input('dropdown-update-layout', 'value')])
def update_layout(layout):
    return {
        'name': layout,
        'animate': True
    }


@app.callback(Output('nodes-table', 'data'),
              [Input('cytoscape-update-layout', 'tapNodeData')])
def update_table_nodes(data):
    if data:
        return characters[characters['Alias'] == data['key']].to_dict('records')


@app.callback(Output('single-relation-table', 'data'),
              [Input('cytoscape-update-layout', 'tapNodeData')])
def update_table_single_relations(data):
    if data:
        return one_char_relations[one_char_relations['Character1'] == data['key']].to_dict('records')


@app.callback(Output('relations-table', 'data'),
              [Input('cytoscape-update-layout', 'tapEdgeData')])
def update_table_relations(data):
    if data:
        from_to = two_chars_relations[two_chars_relations['Character1'] == data['source']][
            two_chars_relations['Character2'] == data['target']]
        return from_to.to_dict('records')


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here


@app.callback(
    dash.dependencies.Output('distances', 'figure'),
    [dash.dependencies.Input('filter-distances', 'value')])
def update_graph(char_id):
    data = []
    if char_id is not None:
        character = 'CCHARACTER' + str(char_id)
        columns = chars_distances.columns
        x = list(range(1, len(columns) + 1))
        x_s = [[] for x in range(len(chars_distances))]
        y_s = [[] for x in range(len(chars_distances))]

        for i, col in enumerate(columns):
            values = chars_distances.loc[character, col]
            for j, value in enumerate(values):
                if value is not None:
                    x_s[j].append(x[i])
                    y_s[j].append(round(value, 2))

        for i, x_char_distance in enumerate(x_s):
            current_name = chars_distances.index[i]
            name = list(chars_embedding[chars_embedding['Alias'] == current_name]['Name'])[0]
            if current_name == character:
                visibility = True
            else:
                visibility = "legendonly"
            data.append(
                go.Scatter(x=x_char_distance, y=y_s[i], name=name, mode='lines+markers+text', visible=visibility))

        layout = go.Layout(xaxis={'title': 'Novel developement'},
                           yaxis={'title': 'Trajectories'})

        return go.Figure(data=data, layout=layout)
    else:
        return go.Figure()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file receives a book (input) '
                                                 'and display the results computed by Code/test_static_dynamic_embedding.py and Code/test_relations_clustering.py.')
    parser.add_argument("input_filename", type=str, help="The input file name \'file.txt\' or folder name")
    try:
        filename = sys.argv[1].split('.')[0]
    except:
        print('Provide an input file! Or try \'-h\' option')
        exit(-1)

    page_one = Page_1(filename)
    page_two = Page_2(filename)

    page_1_layout = page_one.layout_1
    characters = page_one.characters
    one_char_relations = page_one.one_char_relations
    two_chars_relations = page_one.two_chars_relations
    chars_distances = page_two.chars_distances
    chars_embedding = page_two.chars_embedding
    page_2_layout = page_two.layout_2
    app.run_server(debug=True)
