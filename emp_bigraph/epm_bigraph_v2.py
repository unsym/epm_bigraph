"""
unique_nontrivial_EPM_bigraph.py
------------------------------

Enumerate all non-trivial EPM graphs (up to colour-preserving isomorphism)
for given parameters (n_q, n_a) using the igraph library for performance.

Requirements
------------
- Python 3.10+
- igraph (`pip install python-igraph`)

Interface
---------
unique_nontrivial_EPM_bigraph(n_q: int, n_a: int) -> list[igraph.Graph]

Each returned graph has:
    • node attribute 'type' in {'Q', 'A', 'R'}
    • bipartition (Q union A) | R
    • qubit degree == 2
    • auxiliary degree >= 2
    • right-side degree >= 1 (non-trivial)
The list contains exactly one representative from every colour-preserving
isomorphism class.
"""

from itertools import combinations, product
from igraph import Graph

# --------------------------------------------------------------------------- #
#  Helper ­- Canonical signature using igraph canonical_permutation
# --------------------------------------------------------------------------- #
def _canonical_signature(g: Graph) -> str:
    # Map types to colours for canonicalization
    color_map = {'Q': 0, 'A': 1, 'R': 2}
    vertex_colors = [color_map[t] for t in g.vs['type']]
    # Compute canonical permutation
    perm = g.canonical_permutation(color=vertex_colors)
    # Permute to canonical form
    gc = g.permute_vertices(perm)
    # Build a signature from sorted edge list
    edges = sorted(gc.get_edgelist())
    return tuple(edges)


# --------------------------------------------------------------------------- #
#  Generator for *all* non-trivial EPM graphs (no iso-filtering)
# --------------------------------------------------------------------------- #
def _generate_all_nontrivial_EPM_bigraphs(n_q: int, n_a: int, R_min: int=2):
    R_size = n_q + n_a
    # Node index halves:
    # 0..n_q-1        : Q vertices
    # n_q..n_q+n_a-1  : A vertices
    # n_q+n_a..n_q+n_a+R_size-1 : R vertices
    total_vertices = n_q + n_a + R_size
    Q_indices = list(range(n_q))
    A_indices = list(range(n_q, n_q + n_a))
    R_indices = list(range(n_q + n_a, total_vertices))

    # Precompute neighbor options on the right
    two_subsets = list(combinations(R_indices, 2))
    aux_subsets = [
        comb
        for d in range(2, R_size + 1)
        for comb in combinations(R_indices, d)
    ]  # Edge number >=2 for auxiliaries

    # Enumerate choices
    for Q_choice in product(two_subsets, repeat=n_q):
        for A_choice in product(aux_subsets, repeat=n_a):
            # Degree counter for right nodes
            deg_R = {r: 0 for r in R_indices}
            for qnbrs in Q_choice:
                for r in qnbrs:
                    deg_R[r] += 1
            for anbrs in A_choice:
                for r in anbrs:
                    deg_R[r] += 1
            # non-triviality: every right node must have >=R_min edge
            if any(deg < R_min for deg in deg_R.values()):
                continue

            # build the full igraph Graph
            g = Graph(n=total_vertices, directed=False)
            # assign types in node order: Q, then A, then R
            types = ['Q'] * n_q + ['A'] * n_a + ['R'] * R_size
            g.vs['type'] = types

            edges = []
            # qubit edges
            for i, qnbrs in enumerate(Q_choice):
                for r in qnbrs:
                    edges.append((i, r))
            # auxiliary edges
            for j, anbrs in enumerate(A_choice, start=n_q):
                for r in anbrs:
                    edges.append((j, r))
            g.add_edges(edges)
            yield g

# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
def unique_nontrivial_EPM_bigraph(n_q: int, n_a: int):
    """
    Returns one representative per colour-preserving isomorphism class.
    """
    reps = {}
    for g in _generate_all_nontrivial_EPM_bigraphs(n_q, n_a):
        sig = _canonical_signature(g)
        if sig not in reps:
            reps[sig] = g
    return list(reps.values())

# --------------------------------------------------------------------------- #
#  Minimal demo (delete or comment out in production)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo_graphs = unique_nontrivial_EPM_bigraph(3, 2)
    print(f"n_q=2, n_a=1  →  found {len(demo_graphs)} unique EPM bigraphs")
    for idx, g in enumerate(demo_graphs, 1):
        print(f"Graph {idx}: types={g.vs['type']}")
        print(f"  edges={g.get_edgelist()}")
        print()
