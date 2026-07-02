from typing import Callable
import pickle
import numpy as np
import faiss

class FaissIndex:
    """
    This class handles the vectorization of strings and stores them 
    in an indexing structure (FAISS HNSW) for efficient similarity search.
    """

    def __init__(self, vectorize: Callable, metric: int, ef_construction: int=200, M: int=32, ef: int=50):
        """
        Initialize the FAISS index parameters.

        Args:
            vectorize (Callable): A function that takes a string and returns its vector representation.
            ef_construction (int, optional): The depth of the search during index construction for FAISS HNSW. Defaults to 200.
            M (int, optional): The number of bi-directional links created for every new element during HNSW index construction. Defaults to 32.
            ef (int, optional): The depth of the search for FAISS HNSW. Defaults to 50.
        """
        self.vectorize = vectorize
        
        self.metric = metric
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        
        self.entries = None
        self.vectors = None
        self.index = None
    
    def build(self, entries: list[str]):
        """
        Build the FAISS index using the provided entries and vectorization function.
        
        Args:
            entries (list[str]): A list of strings to vectorize and index.
        """
        self.entries = entries
        self.vectors = np.vstack([ self.vectorize(e.strip().lower()) for e in entries ])
        self.index = FaissIndex._build_index(self.vectors, self.metric, self.ef_construction, self.M, self.ef)
        
        return self

    def save(self, filepath: str="index.pkl"):
        """
        Save the vector representations and the FAISS index to a file for later use.

        Args:
            filepath (str, optional): The path to the file where the index should be saved. Defaults to "index.pkl".
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'entries': self.entries, 'vectors': self.vectors}, f)

    def load(self, filepath: str="index.pkl"):
        """
        Load the vector representations from a file and reconstruct the FAISS index.

        Args:
            filepath (str, optional): The path to the file from which the index should be loaded. Defaults to "index.pkl".
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.entries = data['entries']
        self.vectors = data['vectors']
        self.index = FaissIndex._build_index(self.vectors, self.metric, self.ef_construction, self.M, self.ef)
        
        return self

    def lookup(self, queries: list[str], k: int=1):
        """
        Perform a similarity search on the index for a given set of queries.

        Args:
            queries (list[str]): A list of string queries to look up in the index.
            k (int, optional): The number of nearest neighbors to retrieve for each query. Defaults to 1.

        Returns:
            list[tuple[str, list[tuple[str, float]]]]: A list of tuples, where each tuple contains:
                - The original query string
                - A list of `k` nearest neighbors as tuples of (matched_string, distance)
        
        Raises:
            ValueError: If the index has not been built yet.
        """
        if self.index is None:
            raise ValueError("The index has not been built yet. Please call the `build` method before performing lookups.")
        
        query_vectors = np.array([self.vectorize(q) for q in queries], dtype=np.float32)
        distances, labels = self.index.search(query_vectors, k)
        
        results = []
        for query, idx, dists in zip(queries, labels, distances):
            result = [(self.entries[idx], dist) for idx, dist in zip(idx, dists) if idx != -1]
            results.append((query, result))
        
        return results
    
    def _build_index(vectors: np.ndarray, metric: int, ef_construction: int, M: int, ef: int):
        """
        Construct the FAISS HNSW Index based on the built corpus vectors.
        
        Args:
            metric: The FAISS metric to use (e.g. faiss.METRIC_L1).
            ef_construction (int): The index construction depth configuration.
            M (int): The number of bi-directional links created for every new element.
            ef (int): The search depth configuration.
            
        Returns:
            faiss.Index: The constructed FAISS index.
        """ 
        
        dim = vectors.shape[1]
        
        index = faiss.index_factory(dim, f"HNSW{M}", metric)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef
        index.add(vectors)
        
        return index