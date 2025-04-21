"""
unique_nontrivial_EPM_bigraph.py
------------------------------

Enumerate all non-trivial EPM graphs (up to colour-preserving isomorphism)
for given parameters (n_q, n_a).

Requirements
------------
- Python 3.10+
- networkx >= 3.0  (`pip install networkx`)
  - used only for the canonical-form / isomorphism machinery.

Interface
---------
unique_nontrivial_EPM_bigraph(n_q: int, n_a: int) -> list[networkx.Graph]

Each returned graph has
    • node attribute 'type' in {'Q', 'A', 'R'}
    • bipartition (Q union A) | R
    • qubit degree == 2
    • auxiliary degree ≥ 2
    • right-side degree ≥ 1   (=> “non-trivial”)
The list contains exactly one representative from every colour-preserving
isomorphism class.
"""

from itertools import combinations, product
from collections import defaultdict
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


# --------------------------------------------------------------------------- #
#  Helper ­- canonical fingerprint that respects vertex colours
# --------------------------------------------------------------------------- #
def _canonical_signature(G: nx.Graph) -> str:
    """
    Weisfeiler-Lehman hash with the vertex colour stored in node attr 'type'.

    ──  For coloured graphs, WL-hash is injective for *almost* all practical
    cases; if two graphs share the same hash we fall back to a full
    colour-aware isomorphism check to rule out the (rare) collision.  ──
    """
    return weisfeiler_lehman_graph_hash(G, node_attr="type", edge_attr=None)


def _is_coloured_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
    nm = nx.algorithms.isomorphism.categorical_node_match("type", None)
    gm = nx.algorithms.isomorphism.GraphMatcher(G1, G2, node_match=nm)
    return gm.is_isomorphic()


# --------------------------------------------------------------------------- #
#  Generator for *all* non-trivial EPM graphs (no iso-filtering)
# --------------------------------------------------------------------------- #
def _generate_all_nontrivial_EPM_bigraphs(n_q: int, n_a: int):
    R_size = n_q + n_a
    Q_nodes = [f"Q{i}" for i in range(n_q)]
    A_nodes = [f"A{j}" for j in range(n_a)]
    R_nodes = [f"R{k}" for k in range(R_size)]

    # ── Neighbour options on the right ────────────────────────────────────── #
    two_subsets = list(combinations(range(R_size), 2))	# This is WRONG! EPM has colored edges.
    aux_subsets = [
        comb
        for d in range(2, R_size + 1)
        for comb in combinations(range(R_size), d)
    ]  # ≥2 for auxiliaries

    # ── Cartesian product of choices for every left vertex ───────────────── #
    for Q_choice in product(two_subsets, repeat=n_q):
        for A_choice in product(aux_subsets, repeat=n_a):
            # degree counter on the right
            deg_R = [0] * R_size
            for pair in Q_choice:
                for r in pair:
                    deg_R[r] += 1
            for subset in A_choice:
                for r in subset:
                    deg_R[r] += 1
            # non-triviality: every right vertex must be touched
            #if any(d <= 1 for d in deg_R): # and have degree 2
            if any(d <= 0 for d in deg_R):
                continue

            # Build the graph
            G = nx.Graph()
            # add nodes with colour / type
            G.add_nodes_from(Q_nodes, type="Q")
            G.add_nodes_from(A_nodes, type="A")
            G.add_nodes_from(R_nodes, type="R")
            # add edges Q→R
            for qubit, pair in zip(Q_nodes, Q_choice):
                for r_idx in pair:
                    G.add_edge(qubit, R_nodes[r_idx])
            # add edges A→R
            for aux, subset in zip(A_nodes, A_choice):
                for r_idx in subset:
                    G.add_edge(aux, R_nodes[r_idx])

            yield G


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def unique_nontrivial_EPM_bigraph(n_q: int, n_a: int):
    """
    Enumerate *one* representative for every colour-preserving isomorphism
    class of non-trivial EPM graphs with parameters (n_q , n_a).

    Returns
    -------
    list[nx.Graph]   (each with node attr 'type')
    """
    representatives: dict[str, nx.Graph] = {}  # WL-hash → canonical graph

    for G in _generate_all_nontrivial_EPM_bigraphs(n_q, n_a):
        sig = _canonical_signature(G)
        if sig not in representatives:
            representatives[sig] = G
        else:
            # rare WL hash collision - check isomorphism explicitly
            if not _is_coloured_isomorphic(G, representatives[sig]):
                # distinct iso-class; need another key to store it
                representatives[f"{sig}_{len(representatives)}"] = G

    return list(representatives.values())


# --------------------------------------------------------------------------- #
#  Minimal demo (delete or comment out in production)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo_graphs = unique_nontrivial_EPM_bigraph(3, 1)
    print(f"n_q=2, n_a=1  →  found {len(demo_graphs)} unique EPM bigraphs")
    for idx, g in enumerate(demo_graphs, 1):
        print(f"Graph {idx}:")
        print("  Q:", [v for v, d in g.nodes(data=True) if d["type"] == "Q"])
        print("  A:", [v for v, d in g.nodes(data=True) if d["type"] == "A"])
        print("  R:", [v for v, d in g.nodes(data=True) if d["type"] == "R"])
        print("  Edges:", list(g.edges()))
        print()
