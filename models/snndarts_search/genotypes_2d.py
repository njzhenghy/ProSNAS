from collections import namedtuple

Genotype = namedtuple('Genotype_2D', 'cell cell_concat')

PRIMITIVES = [
    'skip_connect',
    'cnn_3x3',
    # 'cnn_5x5',
    # 'none'
    # 'cnn_5x5'
    # 'cnn_7x7'
    ]

PRIMITIVES_SPIKE = [
    'spike3',
    'spike5',
    # 'spike7'
    ]

# PRIMITIVES = [
#     'skip_connect',
#     'snn_3x3',
#     'snn_5x5'
#     ]


