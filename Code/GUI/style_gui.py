colors = {
    'background': '#111111',
    'text': '#000000'
}

text_style = {
            'textAlign': 'center',
            'color': colors['text']
        }

default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': 'data(color)',
            'label': 'data(label)'
        }
    },
    {
        'selector': 'node:selected',
        'style': {
            'overlay-color': 'dark',
            'overlay-opacity': '0.2',
            'overlay-padding': '10'
        }
    },
    {
        'selector': 'edge:selected',
        'style': {
            'overlay-color': 'dark',
            'overlay-opacity': '0.2',
            'overlay-padding': '10'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            # 'curve-style': 'unbundled-bezier',
            # "control-point-weights": 0.1,
            # "control-point-distances": 120,
            'line-color': 'data(color)',
            'target-arrow-shape': 'triangle-backcurve',
            'target-arrow-color': 'data(color)'
        }
    }
]