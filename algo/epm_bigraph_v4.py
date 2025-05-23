"""
Enumerate all unique non-trivial EPM bipartite graphs (up to color-preserving isomorphism)
for given parameters (n_q, n_a), using the igraph library for performance.
"""

from itertools import combinations, combinations_with_replacement
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
    """
    Enumerator for non-trivial Extended Projective Measurement (EPM) bipartite graphs.

    This class generates undirected bipartite graphs with the following properties:
      * The vertex set is partitioned into qubit (Q), ancilla (A), and right-side (R) nodes.
      * The bipartition are between (Q union A) and R.
      * Each Q node is connected to exactly 2 R-nodes.
      * Each A node is connected to ≥2 R nodes.
      * Each R node must be connected to ≥ R_min neighbors (default: 2).
      * Graphs are filtered up to color-preserving isomorphism using canonical labeling.
      * By default, only graphs whose derived directed EPM graph is strongly connected
        are retained (controlled by `strongly_connected_only`).

    The resulting graphs are useful in quantum information contexts such as
    optical graph states, and measurement-based quantum computation.

    Parameters
    ----------
    R_min : int, default=2
        The minimum degree required for each R node in the bipartite graph.
        This ensures non-trivial coupling from Q and A nodes.

    strongly_connected_only : bool, default=True
        If True, filters the output to only include graphs whose corresponding
        directed EPM graph (as defined by EPM_digraph_from_EPM_bipartite_graph_igraph)
        forms a single strongly connected component (SCC).
        This reflects full causal or information-theoretic interdependence in
        the system and is often used as a physical constraint.

    Attributes
    ----------
    node_colors : list[int]
        Color labels for vertices used in canonical isomorphism detection.
        Q = 0, A = 1, R = 2

    node_types : list[str]
        Short type labels for vertices: "Q", "A", or "R"

    node_categories : list[str]
        Full category labels matching what is required by the digraph converter:
        "system_nodes", "ancilla_nodes", and "sculpting_nodes"

    Methods
    -------
    enumerate(n_q: int, n_a: int) -> list[igraph.Graph]
        Returns a list of unique, non-trivial EPM bipartite graphs for the
        given number of qubits and ancillae, optionally filtered by SCC constraint.
    """

    def __init__(self, R_min: int = 2, strongly_connected_only: bool = True):
        self.R_min = R_min
        self.strongly_connected_only = strongly_connected_only
        self.node_ids = None        # id of the node, from 0 to #-1
        self.node_names = None      # name of the node
        self.node_bipartites = None # which partition the node belong to
        self.node_colors = None     # colors/types/categories are the same thing, for different purpose
        self.node_types = None
        self.node_categories = None
        self.n_q = None
        self.n_a = None
        self.Gs = None
        self.num_epm_bigraph_enumerated = 0     # Number of EPM bigraph returned for the test of canonical signature
        self.num_epm_bigraph_canonical = 0      # Number of unique EPM bigraph with canonical signature before SCC filtering
        self.num_epm_bigraph_canonical_nontrivial = 0   # Number of unique EPM bigraph with canonical signature after SCC filtering


    def _attach_node_metadata(self, graphs: list[ig.Graph], attributes: list[str]) -> list[ig.Graph]:
        """
        Attach selected node attributes to each graph in the list.

        Parameters
        ----------
        graphs : list[igraph.Graph]
            List of igraph graph objects to be modified in-place.
        
        attributes : list[str]
            List of attributes to attach. Valid options include:
                "id", "name", "category", "bipartite", "type", "color"

        Returns
        -------
        list[igraph.Graph]
            The same list of graphs with node attributes attached.
        """
        if not attributes:
            return graphs
        for g in graphs:
            n = g.vcount()
            if "id" in attributes:
                g.vs["id"] = self.node_ids[:n]
            if "name" in attributes:
                g.vs["name"] = self.node_names[:n]
            if "category" in attributes:
                g.vs["category"] = self.node_categories[:n]
            if "bipartite" in attributes:
                g.vs["bipartite"] = self.node_bipartites[:n]
            if "type" in attributes:
                g.vs["type"] = self.node_types[:n]
            if "color" in attributes:
                g.vs["color"] = self.node_colors[:n]
        return graphs


    def _bigraph_to_digraph(self, B: ig.Graph, n_q: int, n_a: int) -> ig.Graph:
        """
        Converts an EPM bipartite graph B into a directed graph D over Q ∪ A nodes.
        
        Parameters
        ----------
        B : igraph.Graph
            Undirected bipartite graph with vertices ordered as:
            [0 .. n_q-1]: Q (system), [n_q .. n_q+n_a-1]: A (ancilla), [n_q+n_a .. end]: R (sculpting)

        n_q : int
            Number of Q (system) nodes.

        n_a : int
            Number of A (ancilla) nodes.

        Returns
        -------
        igraph.Graph
            Directed graph D with Q ∪ A nodes (n_q + n_a total).
            For each R-node connected to ≥2 Q/A nodes, creates bidirectional edges among them.
        """
        # Now it consistent give the digraph SCC, different from EPM_digraph_from_EPM_bipartite_graph_igraph().
        n_r = n_q + n_a
        D = ig.Graph(n=n_r, directed=True)

        # For each R-node
        for r in range(n_q + n_a, n_q + n_a + n_r):
            # Get neighbors of r in the bipartite graph
            neighbors = B.neighbors(r)
            # Filter to only include Q or A nodes
            qa_neighbors = [v for v in neighbors if v < n_r]

            # If R connects to multiple Q/A nodes, add directed edges among them
            for i in range(len(qa_neighbors)):
                for j in range(len(qa_neighbors)):
                    if i != j:
                        D.add_edge(qa_neighbors[i], qa_neighbors[j])

        return D


    def _canonical_signature(self, g: ig.Graph) -> tuple:
        """
        Computes a canonical signature for a graph based on vertex color and
        sorted edge structure. Used to identify unique graphs up to 
        color-preserving isomorphism. The return is a tuple of sorted edges
        in the canonical vertex order, used as a hashable key to detect isomorphic graphs.
        """
        perm = g.canonical_permutation(color=self.node_colors)
        gc = g.permute_vertices(perm)
        edges = sorted(gc.get_edgelist())
        return tuple(edges)


    def _generate_all_EPM_bigraphs(self, n_q: int, n_a: int):
        """
        Generator for all EPM bipartite graphs. The yield is an undirected bipartite graph
        with the following degree constraints, but without isomorphism filtering or SCC filtering.

        Each graph has:
        * n_q “qubit” nodes of degree exactly 2
        * n_a “ancilla” nodes of degree >= 2
        * n_r = n_q + n_a “R” nodes, each with degree >= self.R_min

        Nodes are ordered:
        0..n_q-1           : qubit nodes
        n_q..n_q+n_a-1     : ancilla nodes
        n_q+n_a..n_q+n_a+n_r-1 : R nodes
        """
        n_r = n_q + n_a     # total # of R-side nodes
        total_vertices = n_q + n_a + n_r    # total vertices = Q + A + R
        # index ranges for each part
        Q_indices = list(range(n_q))
        A_indices = list(range(n_q, n_q + n_a))
        R_indices = list(range(n_q + n_a, total_vertices))

        # Precompute neighbor options on the right, the edges are treated as unweighted
        two_subsets = list(combinations(R_indices, 2))
        ancilla_subsets = [
            comb
            for d in range(2, n_r + 1)
            for comb in combinations(R_indices, d)
        ]  # Number of edge (or neighbors) >= 2 for ancilla nodes

        # Enumerate every choice of R‐neighbors for Q and A nodes
        for Q_choice in combinations_with_replacement(two_subsets, n_q):
            for A_choice in combinations_with_replacement(ancilla_subsets, n_a):
                # count degrees on R‐side
                deg_R = {r: 0 for r in R_indices}
                for qnbrs in Q_choice:
                    for r in qnbrs:
                        deg_R[r] += 1
                for anbrs in A_choice:
                    for r in anbrs:
                        deg_R[r] += 1
                # non-triviality: every right node must have >= R_min edge, otherwise, skip
                if any(deg < self.R_min for deg in deg_R.values()):
                    continue

                # build the igraph object
                g = ig.Graph(n=total_vertices, directed=False)

                edges = []
                # qubit edges
                for i, qnbrs in enumerate(Q_choice):
                    for r in qnbrs:
                        edges.append((i, r))
                # ancilla edges
                for j, anbrs in enumerate(A_choice, start=n_q):
                    for r in anbrs:
                        edges.append((j, r))
                g.add_edges(edges)
                yield g


    def enumerate_structural(self, n_q: int, n_a: int, attributes: list[str]=[]) -> dict[tuple, ig.Graph]:
        """
        Enumerate unique non-trivial EPM bipartite graphs for given (n_q, n_a).

        This function:
        * Generates all non-trivial EPM graphs using _generate_all_nontrivial_EPM_bigraphs().
        * Deduplicates them up to color-preserving isomorphism using canonical signatures.
        * Optionally filters to retain only those graphs whose corresponding
            directed EPM graph is a single strongly connected component.

        Parameters
        ----------
        n_q : int
            Number of qubit (Q) vertices. Each Q node connects to exactly two R nodes.
        n_a : int
            Number of ancilla (A) vertices. Each A node connects to at least two R nodes.

        Returns
        -------
        dict[tuple, igraph.Graph]
            A dictionary mapping canonical signatures to unique undirected bipartite graphs,
            optionally filtered by SCC constraint if strongly_connected_only is True.
        """
        self.method = "enumerate_structural"
        self.n_q = n_q
        self.n_a = n_a
        self.n_r = n_q + n_a     # total # of R-side nodes
        self.n_total = n_q + n_a + self.n_r    # total vertices = Q + A + R
        # Tag each node with the exact colors/categories/shortname in node order: Q, then A, then R
        self.node_ids = list(range(self.n_total))
        self.node_names = [f"S_{i}" for i in range(n_q)] + [f"A_{j}" for j in range(n_a)] + [f"{k}" for k in range(self.n_r)]
        self.node_bipartites = [0] * (n_q + n_a) + [1] * self.n_r
        self.node_colors = [0] * n_q + [1] * n_a + [2] * self.n_r
        self.node_types = ['Q'] * n_q + ['A'] * n_a + ['R'] * self.n_r
        self.node_categories = ["system_nodes"] * n_q + ["ancilla_nodes"] * n_a + ["sculpting_nodes"] * self.n_r
        self.num_epm_bigraph_enumerated = 0

        self.Gs = {}
        for g in self._generate_all_EPM_bigraphs(n_q, n_a):
            self.num_epm_bigraph_enumerated += 1
            sig = self._canonical_signature(g)
            if sig not in self.Gs:
                self.Gs[sig] = g
        self.num_epm_bigraph_canonical = len(self.Gs)

        if not self.strongly_connected_only:
            return self.Gs

        final_Gs = {}
        for sig, g in self.Gs.items():
            g2 = g.copy()
            g2.vs['category'] = self.node_categories
            g2.es['weight'] = [1] * g2.ecount()
            D = EPM_digraph_from_EPM_bipartite_graph_igraph(g2)
            # D = self._bigraph_to_digraph(g2, n_q, n_a)
            if D.is_connected(mode="STRONG"):
                final_Gs[sig] = g

        self.Gs = final_Gs
        self.num_epm_bigraph_canonical_nontrivial = len(self.Gs)
        return self._attach_node_metadata(self.Gs, attributes)


    def enumerate_colored(self, n_q: int, n_a: int, attributes: list[str] = []) -> dict[tuple, list[ig.Graph]]:
        """ Return the enumeration with colored/weighted edges 
        For each unweighted canonical EPM bigraph:
          * generate all 2**n_q ways to assign weights (1,2)/(2,1) on its Q-edges
            and weight=3 on all A-edges,
          * dedupe *within* that group using VF2 up to weight-preserving iso,
          * attach optional node metadata,
        and return a map: unweighted_signature -> [weighted_graphs...].
        """
        struct_map = self.enumerate_structural(n_q, n_a, attributes=[])
        
        colored_map: dict[tuple, list[ig.Graph]] = {}
        for sig, g_struct in struct_map.items():
            # Precompute each Q-node’s two edge-IDs
            q_edges: list[tuple[int,int]] = []
            for q in range(self.n_q):
                # neighbors in R only:
                rs = sorted(v for v in g_struct.neighbors(q) if v >= self.n_r)
                e1 = g_struct.get_eid(q, rs[0])
                e2 = g_struct.get_eid(q, rs[1])
                q_edges.append((e1, e2))

            base_ecount = g_struct.ecount()
            variants: list[ig.Graph] = []
            # Enumerate all 2^n_q weight assignments, assign (1,2)/(2,1) to each Q-edge pair
            for mask in range(1 << self.n_q):
                g_w = g_struct.copy()
                weights = [3] * base_ecount     # assign default weight=3 to every edge
                for q, (e1, e2) in enumerate(q_edges):
                    if (mask >> q) & 1:
                        weights[e1], weights[e2] = 2, 1
                    else:
                        weights[e1], weights[e2] = 1, 2
                g_w.es["weight"] = weights

                # Deduplicate *within* this structural group
                is_new = True
                for g_prev in variants:
                    if g_w.isomorphic_vf2(g_prev, edge_color1=g_w.es["weight"], edge_color2=g_prev.es["weight"]):
                        is_new = False
                        break
                if is_new:
                    variants.append(g_w)

            if attributes:
                self._attach_node_metadata(variants, attributes)
            colored_map[sig] = variants
        return colored_map




# --------------------------------------------------------------------------- #
#  Minimal demo (delete or comment out in production)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    enumerator = EPMBigraphEnumerator()
    n_q, n_a = 3, 2
    demo_graphs = enumerator.enumerate_structural(n_q, n_a)
    print(f"n_q={n_q}, n_a={n_a}  →  found {len(demo_graphs)} unique non-trivial EPM bigraphs")
    # print(sum([len(v)  for k,v in demo_graphs.items()]))
    # print([len(v)  for k,v in demo_graphs.items()])
    # for k,v in demo_graphs.items():
    #     #print(v)
    #     #print(len(v))
    #     for g in v: 
    #         print(list(zip(g.get_edgelist(), g.es["weight"])))
    for (idx,(h, g)) in enumerate(demo_graphs.items()):
        #print(f"Graph {idx}: types={g.vs['type']}")
        print(f"Graph {idx}")
        print(f"  edges={g.get_edgelist()}")
        print()
