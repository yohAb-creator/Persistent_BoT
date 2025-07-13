import numpy as np
import pickle
from typing import Dict, List

# --- Dependency Imports ---
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: 'sentence-transformers' not found. Will use simulated embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import gudhi as gd

    GUDHI_AVAILABLE = True
except ImportError:
    print("Warning: 'gudhi' not found. Persistence analysis will be unavailable.")
    gd = None
    GUDHI_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: 'matplotlib' not found. Visualization will be unavailable.")
    MATPLOTLIB_AVAILABLE = False

try:
    import kmapper as km
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    import networkx as nx

    MAPPER_AVAILABLE = True
except ImportError:
    print("Warning: 'kmapper' or its dependencies not found. Choke point analysis will be unavailable.")
    MAPPER_AVAILABLE = False

from common_structures import ThoughtNode


class ThoughtSpaceAnalyzer:
    """
    Performs TDA on a thought space, including persistence analysis
    and Mapper-based choke point identification.
    """

    def __init__(self, thought_space: Dict[int, ThoughtNode], embedding_model_name='all-MiniLM-L6-v2'):
        if not isinstance(thought_space, dict) or not all(isinstance(v, ThoughtNode) for v in thought_space.values()):
            raise TypeError("Input 'thought_space' must be a dictionary of ThoughtNode objects from common_structures.")

        self.thought_space = thought_space
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)
        self.point_cloud = None

    def _initialize_embedding_model(self, model_name):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        try:
            print(f"Loading embedding model: {model_name}...")
            return SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading sentence-transformer model: {e}. Falling back to simulated embeddings.")
            return None

    def _ensure_embeddings_exist(self):
        """A helper method to generate embeddings if they haven't been already."""
        if self.point_cloud is not None:
            return

        # Correctly access the 'text' attribute from the ThoughtNode
        all_thought_texts = [node.text for node in self.thought_space.values()]
        if self.embedding_model and all_thought_texts:
            print(f"Generating real semantic embeddings for {len(all_thought_texts)} thoughts...")
            self.point_cloud = self.embedding_model.encode(all_thought_texts, show_progress_bar=True)
        else:
            print(f"Generating simulated embeddings for {len(all_thought_texts)} thoughts...")
            dimension = 384
            np.random.seed(42)
            self.point_cloud = np.random.rand(len(all_thought_texts), dimension)

    def analyze_topology(self, max_edge_length=1.0, h0_persistence_threshold=0.2, h1_persistence_threshold=0.1):
        """Analyzes the thought space using Persistent Homology on an expanding Rips complex."""
        print("\n\n===================================")
        print("      PERSISTENT HOMOLOGY ANALYSIS")
        print("===================================")

        if not GUDHI_AVAILABLE:
            print("Analysis aborted: Gudhi library not found.")
            return

        self._ensure_embeddings_exist()

        print(f"\nBuilding Rips complex (distance threshold: {max_edge_length})...")
        rips_complex = gd.RipsComplex(points=self.point_cloud, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence_pairs = simplex_tree.persistence()

        self._interpret_and_display_persistence(persistence_pairs, max_edge_length, h0_persistence_threshold,
                                                h1_persistence_threshold)

    def _interpret_and_display_persistence(self, persistence_pairs, max_edge_length, h0_threshold, h1_threshold):
        """Filters, interprets, and visualizes the persistence results."""
        if not persistence_pairs:
            print("No topological features were found.")
            return

        persistent_h0 = [p for p in persistence_pairs if p[0] == 0 and (p[1][1] - p[1][0]) > h0_threshold]
        persistent_h1 = [p for p in persistence_pairs if p[0] == 1 and (p[1][1] - p[1][0]) > h1_threshold]

        print("\n--- Filtered TDA Results ---")
        print(f"Found {len(persistent_h0)} significant H0 features (Conceptual Clusters)")
        print(f"Found {len(persistent_h1)} significant H1 features (Reasoning Loops)")

        if MATPLOTLIB_AVAILABLE:
            gd.plot_persistence_diagram(persistence=persistence_pairs, legend=True)
            plt.title("Persistence Diagram of Thought Space")
            plt.show()

    def find_choke_points(self, n_intervals=15, perc_overlap=0.4, dbscan_eps=0.5, dbscan_min_samples=3):
        """Identifies central "choke point" thoughts using the Mapper algorithm."""
        print("\n\n===================================")
        print("      MAPPER CHOKE POINT ANALYSIS")
        print("===================================")

        if not MAPPER_AVAILABLE:
            print("Analysis aborted: kmapper or its dependencies not found.")
            return {}

        if not self.thought_space or len(self.thought_space) < dbscan_min_samples:
            print("Analysis aborted: Not enough thoughts for Mapper analysis.")
            return {}

        self._ensure_embeddings_exist()

        print("Applying filter function (PCA)...")
        lens = PCA(n_components=2).fit_transform(self.point_cloud)
        mapper = km.KeplerMapper(verbose=0)
        cover = km.Cover(n_cubes=n_intervals, perc_overlap=perc_overlap)
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')

        print("Running Mapper algorithm...")
        mapper_graph_data = mapper.map(lens=lens, X=self.point_cloud, cover=cover, clusterer=clusterer)

        if not mapper_graph_data or not mapper_graph_data.get('nodes'):
            print("Mapper graph has no nodes. Try adjusting parameters.")
            return {}

        print("\nAnalyzing Mapper graph structure...")
        nx_graph = km.adapter.to_nx(mapper_graph_data)
        print(f"Mapper graph created with {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges.")

        choke_points = []
        if nx_graph.number_of_nodes() > 0:
            nodes_by_degree = sorted(dict(nx_graph.degree()).items(), key=lambda item: item[1], reverse=True)
            print("\nTop 5 Potential Choke Points (Hubs in the Thought Graph):")
            ordered_thoughts = sorted(list(self.thought_space.values()), key=lambda t: t.id)

            for i, (node_id_str, degree) in enumerate(nodes_by_degree[:5]):
                member_indices = mapper_graph_data['nodes'].get(node_id_str, [])
                if not member_indices: continue

                # Correctly access the 'score' attribute from the ThoughtNode
                avg_score = np.mean([ordered_thoughts[idx].score for idx in member_indices])
                representative_text = ordered_thoughts[member_indices[0]].text

                print(f"  Hub {i + 1} (Degree: {degree}, Avg Score: {avg_score:.2f}): '{representative_text[:60]}...'")
                choke_points.append({
                    "hub_id": node_id_str,
                    "degree": degree,
                    "avg_score": avg_score,
                    "member_indices": member_indices
                })
        else:
            print("No connections found in Mapper graph.")

        # CORRECTED: Return the choke points under the expected key
        return {
            'choke_points': choke_points,
            'mapper_graph_data': mapper_graph_data,
            'nx_graph': nx_graph
        }
