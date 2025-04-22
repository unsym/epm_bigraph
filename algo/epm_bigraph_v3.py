"""
unique_nontrivial_EPM_bigraph.py
------------------------------

Enumerate all non-trivial EPM graphs (up to colour-preserving isomorphism)
for given parameters (n_q, n_a) using the igraph library for performance.

Requirements
------------
- Python 3.10+
- igraph (`pip install python-igraph`)
- numpy (`pip install numpy`)

Interface
---------
unique_nontrivial_EPM_bigraph(n_q: int, n_a: int) -> list[igraph.Graph]

Each returned graph has:
    • node attribute 'type' in {'Q', 'A', 'R'}
    • bipartition (Q union A) | R
    • qubit degree == 2
    • auxiliary degree >= 2
    • right-side degree >= 2 (non-trivial criteria)
The list contains exactly one representative from every colour-preserving
isomorphism class *and* whose derived directed EPM graph is a single
strongly connected component (non-trivial criteria).
"""

from itertools import combinations, product
import numpy as np
import igraph as ig

def EPM_digraph_from_EPM_bipartite_graph_igraph(B: ig.Graph) -> ig.Graph:
    """
    Converts an EPM bipartite graph (B) from igraph format to a directed graph (D).

    This function classifies nodes of the input bipartite graph B into 'system',
    'ancilla', and 'sculpting' categories, then generates directed edges
    between nodes according to specific rules, returning a new directed graph D.
    It uses adjacency matrix operations to determine edge direction and weights.

    Parameters:
    -----------
    B : igraph.Graph
        The EPM bipartite graph object to convert.
        Must contain 'category' node attribute ('system_nodes', 'ancilla_nodes',
        'sculpting_nodes') and 'weight' edge attribute.

    Returns:
    --------
    igraph.Graph
        The generated directed EPM graph object (D).
        Contains 'category' and 'name' node attributes, and 'weight' edge attribute.

    Raises:
    -------
    KeyError:
        If required attributes 'category' or 'weight' are missing in graph B.
    ValueError:
        If the graph structure is inconsistent or validation fails.
    Exception:
        For any other unexpected errors during conversion.
    """
    try:
        # Identify system, ancilla, and sculpting nodes
        system_nodes = [v.index for v in B.vs if v['category'] == "system_nodes"]
        ancilla_nodes = [v.index for v in B.vs if v['category'] == "ancilla_nodes"]
        sculpting_nodes = [v.index for v in B.vs if v['category'] == "sculpting_nodes"]

        num_system = len(system_nodes)
        num_ancilla = len(ancilla_nodes)
        num_sculpting = len(sculpting_nodes)
        num_total_bipartite = B.vcount()
        num_total_digraph = num_system + num_ancilla # Number of nodes in the resulting directed graph

        # --- Validations ---
        if num_total_bipartite != num_system + num_ancilla + num_sculpting:
            raise ValueError("Sum of node counts does not match the total number of nodes in the bipartite graph.")
        # Assuming the number of sculpting nodes should match the number of system + ancilla nodes
        # Adjust this logic if the EPM definition used is different
        if num_sculpting != num_total_digraph:
            print(f"Warning: Sculpting node count ({num_sculpting}) does not match system+ancilla count ({num_total_digraph}). Ensure this is intended.")
            # Optionally, raise ValueError if this should be strictly enforced

        # Prepare node order: system, ancilla, sculpting
        ordered_vertices = system_nodes + ancilla_nodes + sculpting_nodes

        # Calculate adjacency matrix (with weights)
        # Ensure it's a dense numpy array
        # Use get_adjacency_sparse() if memory is a concern for large graphs
        adj_matrix_B = np.array(B.get_adjacency(attribute="weight").data)

        # Create reordered adjacency matrix using NumPy indexing
        # Size: (num_total_bipartite, num_total_bipartite)
        reordered_adj_matrix = adj_matrix_B[np.ix_(ordered_vertices, ordered_vertices)]

        # Extract the relevant submatrix for the directed graph
        # Rows: system + ancilla nodes (indices 0 to num_total_digraph-1 in reordered matrix)
        # Columns: sculpting nodes (indices num_total_digraph to num_total_bipartite-1 in reordered matrix)
        # adj_matrix_D_sub[i, k] is the weight between the i-th (sys/anc) node and the k-th sculpting node
        adj_matrix_D_sub = reordered_adj_matrix[:num_total_digraph, num_total_digraph:]

        # Initialize the directed graph D
        D = ig.Graph(n=num_total_digraph, directed=True)

        # Set node attributes for D
        categories = ["system_nodes"] * num_system + ["ancilla_nodes"] * num_ancilla
        node_names = [f"S_{i}" for i in range(num_system)] + [f"A_{i}" for i in range(num_ancilla)]
        D.vs["category"] = categories
        D.vs["name"] = node_names

        # Add directed edges
        edges = []
        weights = []

        # Edge direction in D is from sculpting node index 'j' to system/ancilla node index 'i'
        for i in range(num_total_digraph):  # Index for system/ancilla nodes in D
            for k in range(num_sculpting):  # Index for sculpting nodes in the submatrix column
                # Map sculpting node index 'k' to potential source node 'j' in D
                j = k  # Assuming direct mapping; adjust if needed based on EPM definition
                if j < num_total_digraph:  # Ensure source node index 'j' is valid for D
                    weight = adj_matrix_D_sub[i, k]
                    if weight != 0:
                        # Direction is from j to i (consistent with original code logic)
                        edges.append((j, i))
                        weights.append(weight)
                else:
                    # This case might occur if num_sculpting > num_total_digraph, which was warned about earlier
                    print(f"Warning: Sculpting node index {k} maps to digraph index {j}, which is out of bounds ({num_total_digraph}). Skipping potential edge.")

        if edges:
            D.add_edges(edges)
            D.es["weight"] = weights

        return D

    except KeyError as e:
        print(f"Error: Missing required attribute in input graph B: {e}")
        # Re-raise the exception to signal the error upwards
        raise
    except ValueError as e:
        print(f"Error: Graph structure validation failed: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred during digraph conversion: {e}")
        raise




class EPMBigraphEnumerator:
    """Enumerator for unique nontrivial EPM bigraphs."""

    def __init__(self, R_min: int = 2):
        """
        R_min: minimum degree required on each right-side node
        """
        self.R_min = R_min
        self.node_categories = None
        self.short_categories = None

    # ----------------------------------------------------------------------- #
    #  Helper ­- Canonical signature using igraph canonical_permutation
    # ----------------------------------------------------------------------- #
    def _canonical_signature(self, g: ig.Graph) -> tuple:
        # Compute canonical permutation
        perm = g.canonical_permutation(color=self.node_colors)
        # Permute to canonical form
        gc = g.permute_vertices(perm)
        # Build a signature from sorted edge list
        edges = sorted(gc.get_edgelist())
        return tuple(edges)     # This can be used as the key for the unique canonical representation of all isomorphic graph

    # ----------------------------------------------------------------------- #
    #  Generator for *all* non-trivial EPM graphs (no iso-filtering)
    # ----------------------------------------------------------------------- #
    def _generate_all_nontrivial_EPM_bigraphs(self, n_q: int, n_a: int):
        R_size = n_q + n_a
        # Node index halves:
        # 0..n_q-1        : Q vertices
        # n_q..n_q+n_a-1  : A vertices
        # n_q+n_a..n_q+n_a+R_size-1 : R vertices
        total_vertices = n_q + n_a + R_size
        Q_indices = list(range(n_q))
        A_indices = list(range(n_q, n_q + n_a))
        R_indices = list(range(n_q + n_a, total_vertices))
        # tag each node with the exact colors/categories/shortname in node order: Q, then A, then R
        self.node_colors = [0] * n_q + [1] * n_a + [2] * R_size
        self.node_types = ['Q'] * n_q + ['A'] * n_a + ['R'] * R_size
        self.node_categories = ["system_nodes"] * n_q + ["ancilla_nodes"] * n_a + ["sculpting_nodes"] * R_size

        # Precompute neighbor options on the right, the edges are treated as uncolored/unweighted
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
                if any(deg < self.R_min for deg in deg_R.values()):
                    continue

                # build the full igraph Graph
                g = ig.Graph(n=total_vertices, directed=False)

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

    # ----------------------------------------------------------------------- #
    #  Public API
    # ----------------------------------------------------------------------- #
    def enumerate(self, n_q: int, n_a: int):
        """
        Returns one representative per colour-preserving isomorphism class
        whose corresponding directed EPM graph is strongly connected, with degree of all R nodes >=2.
        """
        reps = {}
        for g in self._generate_all_nontrivial_EPM_bigraphs(n_q, n_a):
            sig = self._canonical_signature(g)
            if sig not in reps:
                reps[sig] = g

        # Filter to only those whose digraph has exactly one strongly connected component
        final_reps = {}
        for sig, g in reps.items():
            g2 = g.copy()
            g2.vs['category'] = self.node_categories
            # give every edge a weight attribute so get_adjacency("weight") works
            g2.es['weight'] = [1] * g2.ecount()
            # now convert and test strong connectivity
            D = EPM_digraph_from_EPM_bipartite_graph_igraph(g2)
            if D.is_connected(mode="STRONG"):
                final_reps[sig] = g

        return list(final_reps.values())


# --------------------------------------------------------------------------- #
#  Minimal demo (delete or comment out in production)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    enumerator = EPMBigraphEnumerator()
    demo_graphs = enumerator.enumerate(3, 2)
    print(f"n_q=3, n_a=2  →  found {len(demo_graphs)} unique non-trivial EPM bigraphs")
    for idx, g in enumerate(demo_graphs, 1):
        print(f"Graph {idx}: types={g.vs['type']}")
        print(f"  edges={g.get_edgelist()}")
        print()
