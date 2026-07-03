import numpy as np
import faiss
from faiss_index import FaissIndex

class VecFuzz:
    """
    VecFuzz is a class that provides functionality for vectorizing strings and performing fuzzy matching using FAISS HNSW indexing.
    It allows for efficient similarity search and retrieval of nearest neighbors based on vector representations of strings.
    """

    def __init__(self, chars: str="abcdefghijklmnopqrstuvwxyz0123456789-̧ '. ", ef_construction: int=200, M: int=32, ef: int=50):
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

    def index(self, entries: list[str]):
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
    
    def load(self, filepath = "index.zip"):
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
        3. Preceding characters proximity-weights
        4. Succeeding characters proximity-weights
        
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
        
        vec_pre = np.zeros(self._chars_len, dtype=np.float32)     # Vector based on preceding chars
        vec_suc = np.zeros(self._chars_len, dtype=np.float32)     # Vector based on succeeding chars
        
        for i, ch in enumerate(word, start=1):
            if ch in self._char_idx:
                idx = self._char_idx[ch]
                        
                for j in range(1, i):
                    pos = j / w_len
                    weight = (pos + BOOST) * (DECAY ** (i - j))
                    vec_pre[idx] += weight / w_len

                for j in range(i + 1, w_len + 1):
                    pos = (w_len - j) / w_len
                    weight = (pos + BOOST) * (DECAY ** (j - i))
                    vec_suc[idx] += weight / w_len

        vector = np.concatenate([vec_frq, vec_pos, vec_pre, vec_suc])
        return vector