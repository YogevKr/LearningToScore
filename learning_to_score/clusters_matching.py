import networkx as nx
import numpy as np
from networkx import Graph
from networkx.algorithms import bipartite


def match(y: np.ndarray, pred: np.ndarray):
    # Initialise the graph
    weighted_b_pred: Graph = nx.Graph()

    # Add nodes with the node attribute "bipartite"
    y_nodes = np.unique(y)
    shift_pred_label = max(len(y_nodes), 100)
    pred_nodes = np.unique(pred) + shift_pred_label
    weighted_b_pred.add_nodes_from(y_nodes, bipartite=0)
    weighted_b_pred.add_nodes_from(pred_nodes, bipartite=1)

    # Add edges with weight set to error prob
    for prd_i in pred_nodes:
        pred_01 = pred == (prd_i - shift_pred_label)
        for y_i in y_nodes:
            acc = sum((y == y_i) & pred_01)
            weighted_b_pred.add_edge(
                prd_i, y_i, weight=1 - acc / np.count_nonzero(y == y_i)
            )

    # Find minimum_weight_full_matching
    matching = bipartite.matching.minimum_weight_full_matching(
        weighted_b_pred, top_nodes=y_nodes, weight="weight"
    )

    # Remap pred and calc accuracy
    new_pred = pred.copy()
    for prd_i in pred_nodes:
        new_pred[pred == (prd_i - shift_pred_label)] = matching.get(prd_i, 0)

    return weighted_b_pred, matching, new_pred
