from typing import Callable
import pickle
import io
import zipfile

import numpy as np
import faiss


class VecFuzz:
    """
    VecFuzz is a class that provides functionality for vectorizing strings and performing fuzzy matching using FAISS HNSW indexing.
    It allows for efficient similarity search and retrieval of nearest neighbors based on vector representations of strings.
    """

    def __init__(self, chars: str="aàbcçdeéèêëfghiïjklmnñoöpqrstuvwxyz0123456789-̧ '. ", ef_construction: int=200, M: int=32, ef: int=50):
        """
        Initialize the VecFuzz instance with a set of valid characters for vectorization.

        Args:
            chars (str): A string containing all valid characters for vectorization.
            ef_construction (int, optional): The depth of the search during index construction for FAISS HNSW. Defaults to 200.
            M (int, optional): The number of bi-directional links created for every new element during HNSW index construction. Defaults to 32.
            ef (int, optional): The depth of the search for FAISS HNSW. Defaults to 50.
        """
        
        self._chars = chars
        self._chars_len = len(chars)
        self._char_idx = {c: i for i, c in enumerate(chars)}
        
        self._ef_construction = ef_construction
        self._M = M
        self._ef = ef

        self._index = None

    def build_index(self, entries: list[str]):
        """
        Create a FAISS index from a list of string entries.

        Args:
            entries (list[str]): A list of strings to vectorize and index.
            ef_construction (int, optional): The depth of the search during index construction for FAISS HNSW. Defaults to 200.
            M (int, optional): The number of bi-directional links created for every new element during HNSW index construction. Defaults to 32.
            ef (int, optional): The depth of the search for FAISS HNSW. Defaults to 50.
        """
        self._index = FaissIndex(self.vectorize, faiss.METRIC_L1, self._ef_construction, self._M, self._ef).build(entries)
        return self._index
    
    def load_index(self, filepath = "index.zip"):
        """
        Load a previously saved FAISS index from a file.

        Args:
            filepath (str, optional): The path to the file from which the index should be loaded. Defaults to "index.zip".
        """
        self._index = FaissIndex(self.vectorize, faiss.METRIC_L1, self._ef_construction, self._M, self._ef).load(filepath)
        return self._index

    def vectorize(self, word: str):
        """
        Convert a given word into an overlapping positional, count, and neighbor-based representation float vector.

        It generates a concatenated vector with 4 distinct sub-vectors:
        1. Character frequencies
        2. Average character position
        3. Preceding characters influence
        4. Succeeding characters influence
        
        All sub-vectors are normalized by the length of the word to ensure scale invariance.

        Args:
            word (str): The string to vectorize.

        Returns:
            np.ndarray: A numpy array of type float32 representing the word.
        """
        word = word.strip().lower()
        w_len = len(word)
        
        if w_len == 0:
            raise ValueError("The input word is empty and cannot be vectorized.")
        
        vec_frq = np.zeros(self._chars_len, dtype=np.float32)     # Vector based on char frequency
        vec_pos  = np.zeros(self._chars_len, dtype=np.float32)    # Vector based on char position

        for i, ch in enumerate(word, start=1):
            if ch in self._char_idx:
                idx = self._char_idx[ch]
                vec_frq[idx] += 1 / w_len
                vec_pos[idx]  += i / w_len

        # Context-based vectors
        DECAY = 0.9     # Reduces the influence of farther characters
        BOOST = 3.5     # Amplifies the influence of neighboring characters
        
        vec_pre = np.zeros(self._chars_len, dtype=np.float32)     # Vector based on preceding chars influence
        vec_suc = np.zeros(self._chars_len, dtype=np.float32)     # Vector based on succeeding chars influence
        
        for i, ch in enumerate(word):
            if ch in self._char_idx:
                idx = self._char_idx[ch]

                for j in range(i):
                    pos = (j + 1) / w_len
                    dist = i - j
                    
                    weight = (pos + BOOST) * (DECAY ** dist)
                    vec_pre[idx] += weight / w_len

                for j in range(i + 1, w_len):
                    pos = (w_len - j) / w_len
                    dist = j - i
                    
                    weight = (pos + BOOST) * (DECAY ** dist)
                    vec_suc[idx] += weight / w_len

        # TODO: In a seperacte vectorization function, consider adding additional phonetic and linguistic features to the vector representation of the word.
        # - Consider adding a voyel/consonant ratio vector to capture phonetic characteristics of the word.
        # - Consider adding an International Phonetic Alphabet (IPA) vector to capture pronunciation features of the word.
        
        vector = np.concatenate([vec_frq, vec_pos, vec_pre, vec_suc])
        return vector


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
        self.vectors = np.vstack([ self.vectorize(e) for e in entries ])
        self.index = FaissIndex._build_index(self.vectors, self.metric, self.ef_construction, self.M, self.ef)
        
        return self

    def save(self, filepath: str="index.zip"):
        """
        Save the vector representations and the FAISS index to a file for later use.

        Args:
            filepath (str, optional): The path to the file where the index should be saved. Defaults to "index.zip".
        """
        # Serialize the FAISS index to an in-memory byte buffer
        faiss_buffer = io.BytesIO()
        writer = faiss.PyCallbackIOWriter(faiss_buffer.write)
        faiss.write_index(self.index, writer)
        
        # Serialize the text data and vectors via pickle
        metadata_buffer = io.BytesIO()
        pickle.dump({'entries': self.entries, 'vectors': self.vectors}, metadata_buffer)
        
        # Zip them together into the single destination file
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('metadata.pkl', metadata_buffer.getvalue())
            zf.writestr('faiss.index', faiss_buffer.getvalue())

    def load(self, filepath: str="index.zip"):
        """
        Load the vector representations from a file and reconstruct the FAISS index.

        Args:
            filepath (str, optional): The path to the file from which the index should be loaded. Defaults to "index.zip".
        """
        with zipfile.ZipFile(filepath, 'r') as zf:
            # Load the text data and vectors
            metadata_bytes = zf.read('metadata.pkl')
            data = pickle.loads(metadata_bytes)
            self.entries = data['entries']
            self.vectors = data['vectors']
            
            # 2. Load the compiled FAISS index
            faiss_bytes = zf.read('faiss.index')
            reader = faiss.PyCallbackIOReader(io.BytesIO(faiss_bytes).read)
            self.index = faiss.read_index(reader)
        
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