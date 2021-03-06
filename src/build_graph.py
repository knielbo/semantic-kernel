import os
import argparse
import numpy as np
from datetime import date
import matplotlib.pyplot as plt

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from community import community_louvain
from util import bcolors

def dist_prune(DELTA, prune=True):
    """ transform similarity matrix to distance matrix
    - prune matrix by removing edges that have a distance larger
        than condition cond (default mean distance)
    """
    w = np.max(DELTA)
    DELTA = np.abs(DELTA - w)
    np.fill_diagonal(DELTA, 0.)
    if prune:
        cond = np.mean(DELTA)# + np.std(DELTA)
        for i in range(DELTA.shape[0]):
            for j in range(DELTA.shape[1]):
                val = DELTA[i, j]
                if val > cond:
                    DELTA[i, j] = 0.
                else:
                    DELTA[i, j] = DELTA[i, j]

    return DELTA


def gen_graph(DELTA, labels, figname="nucleus_graph.png"):
    """ generate graph and plot from DELTA distance matrix
    - labels is list of node labels corresponding to columns/rows in DELTA
    """
    DELTA = DELTA * 10  # scale
    dt = [("len", float)]
    DELTA = DELTA.view(dt)

    #  Graphviz
    G = nx.from_numpy_matrix(DELTA)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))
    pos = graphviz_layout(G)

    G = nx.drawing.nx_agraph.to_agraph(G)

    G.edge_attr.update(color="blue", width="2.0")
    G.node_attr.update(color="red", style="filled")
    G.draw(
        os.path.join("fig", "graph_graphviz.png"),
        format="png", prog="neato"
        )

    G = nx.from_numpy_matrix(DELTA)
    G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), labels)))
    nx.draw(G, pos=pos, with_labels=True, node_size=100)
    plt.savefig(os.path.join("fig", "graph_nx.png"))
    plt.close()

    np.random.seed(seed=1234)
    parts = community_louvain.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]

    plt.figure(figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')
    plt.axis("off")
    nx.draw_networkx(
        G, pos=pos, cmap=plt.get_cmap("Pastel1"), node_color=values,
        node_size=500, font_size=12, width=1, font_weight="bold",
        font_color="k", alpha=1, edge_color="gray"
        )

    plt.tight_layout()
    plt.savefig(figname)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="[INFO] build graph from nodes and edges")
    ap.add_argument("-s", "--seed", required=True, help="path to seed file")
    args = vars(ap.parse_args())


    print(f"{bcolors.OKGREEN}[INFO] drawing graph ...{bcolors.ENDC}")
    with open(args["seed"], "r") as fobj:
        seeds = fobj.read().splitlines()
    



    delta = np.loadtxt(
        os.path.join("mdl", "edges.dat"), delimiter=","
        )

    DELTA = dist_prune(delta, prune=True)

    with open(os.path.join("mdl", "nodes.dat"), "r") as f:
        labels = f.read().split("\n")
    labels = labels[:-1]
    
    outname = os.path.join("fig", f"g_cluster-{'-'.join(seeds).lower()}-{date.today()}.png")
    gen_graph(DELTA, labels, figname=outname)

    print(f"{bcolors.OKGREEN}[INFO] writing graph to: {bcolors.ENDC}{bcolors.WARNING}{outname}{bcolors.ENDC}")


if __name__ == '__main__':
    main()