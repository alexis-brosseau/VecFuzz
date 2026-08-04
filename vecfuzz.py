from typing import Callable
import pickle
import io
import os
import zipfile

import numpy as np
import faiss


class VecFuzz:
    """
    VecFuzz is a class that provides functionality for vectorizing strings and performing fuzzy matching using FAISS HNSW indexing.
    It allows for efficient similarity search and retrieval of nearest neighbors based on vector representations of strings.
    """

    def __init__(self, chars: str="aàbcçdeéèêëfghiïjklmnñoöpqrstuvwxyz0123456789-̧ '. ", ef_construction: int=200, M: int=32, ef: int=64, num_threads: int | None = None):
        """
        Initialize the FAISS index parameters.

        Args:
            chars (str): A string containing all valid characters for vectorization.
            ef_construction (int, optional): The depth of the search during index construction for FAISS HNSW. Defaults to 200.
            M (int, optional): The number of bi-directional links created for every new element during HNSW index construction. Defaults to 32.
            ef (int, optional): The depth of the search for FAISS HNSW. Defaults to 50.
            num_threads: Number of thread used to build the index. Default to the maximum available on the system, or 1 if the system cannot determine the number of available threads.
        """
        
        self._chars = chars
        self._chars_len = len(chars)
        self._char_idx = {c: i for i, c in enumerate(chars)}
        
        self.metric = faiss.METRIC_L1
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self.num_threads = num_threads or os.cpu_count() or 1
        
        self.entries = None
        self.vectors = None
        self.index = None
        
    def vectorize(self, word: str):
        """
        Warning: This method is not optimized for batch processing and may be slow for large datasets. 
        Use vectorize_batch() for better performance. 
        This method is provided for convenience.
        
        Convert a given word into a concatenated positional/frequency representation.

        Produces 5 sub-vectors, each of length self._chars_len (except the phase
        block, which is 2*K*self._chars_len):

        1. Character frequency: count of each character, normalized by word length.
        2. Preceding-position density: for each character, a sum of positions that came before it.
        3. Succeeding-position density: for each character, a sum of positions that came after it.
        4. Phase-encoded position: sinusoidal expansion of each character's position(s) at a few frequencies (cos/sin pairs per band).
        5. Adjacency-hash: fixed-size hashed counts of (char, next-char) bigrams, normalized by word length. Captures local ordering.
        All sub-vectors are normalized by word length for scale invariance.

        Args:
            word (str): The string to vectorize.

        Returns:
            np.ndarray: A numpy array of type float32 representing the word.
        """
        word = word.strip().lower()
        w_len = len(word)

        if w_len == 0:
            raise ValueError("The input word is empty and cannot be vectorized.")

        vec_frq = np.zeros(self._chars_len, dtype=np.float32)  # char frequency
        vec_pre = np.zeros(self._chars_len, dtype=np.float32)  # preceding-position density
        vec_suc = np.zeros(self._chars_len, dtype=np.float32)  # succeeding-position density

        PHASE_FREQS = [1.0, 2.0] # frequency bands (in units of pi) for phase encoding
        num_bands = len(PHASE_FREQS)
        vec_phase = np.zeros(2 * num_bands * self._chars_len, dtype=np.float32) # phase-encoded position

        ADJ_DIM = 64
        vec_adj = np.zeros(ADJ_DIM, dtype=np.float32)  # adjacency-hash of (char, next-char) pairs

        for i, ch in enumerate(word):
            pos = (i + 1) / w_len

            if ch in self._char_idx:
                idx = self._char_idx[ch]

                vec_frq[idx] += 1 / w_len
                vec_pre[idx] += i               * (i + 1)       / (2 * w_len ** 2)
                vec_suc[idx] += (w_len - 1 - i) * (w_len - i)   / (2 * w_len ** 2)

                for k, freq in enumerate(PHASE_FREQS):
                    theta = freq * np.pi * pos
                    vec_phase[(2 * k) * self._chars_len + idx] += np.cos(theta) / w_len
                    vec_phase[(2 * k + 1) * self._chars_len + idx] += np.sin(theta) / w_len

                if i + 1 < w_len:
                    next_ch = word[i + 1]
                    if next_ch in self._char_idx:
                        next_idx = self._char_idx[next_ch]
                        pair_id = idx * self._chars_len + next_idx
                        bucket = pair_id % ADJ_DIM
                        vec_adj[bucket] += 1 / w_len

        vector = np.concatenate([vec_frq, vec_pre, vec_suc, vec_phase, vec_adj])
        return vector
    
    def vectorize_batch(self, words: list[str]) -> np.ndarray:
        """
        Batch version of vectorize(). Used for building the index and looking up queries.
        
        Returns:
            np.ndarray: A 2D numpy array of shape (len(words), vector_length) containing the vector representations of the input words.
        """
        words = [w.strip().lower() for w in words]

        PHASE_FREQS = [1.0, 2.0]
        ADJ_DIM = 64
        chars_len = self._chars_len

        n = len(words)
        lens = np.array([len(w) for w in words], dtype=np.int64)
        max_len = int(lens.max())

        # char -> idx lookup, -1 for unknown/padding
        idx_all = np.full((n, max_len), -1, dtype=np.int64)
        for r, w in enumerate(words):
            for c, ch in enumerate(w):
                idx_all[r, c] = self._char_idx.get(ch, -1)

        i = np.arange(max_len)[None, :]          # (1, max_len)
        w_len_col = lens[:, None]                # (n, 1)
        mask = (i < w_len_col) & (idx_all >= 0)  # valid, in-vocab positions

        word_id, i_valid = np.nonzero(mask)      # flat lists of (row, col) for valid entries
        idx = idx_all[word_id, i_valid]
        wl = lens[word_id].astype(np.float32)
        pos = (i_valid + 1) / wl

        flat = word_id * chars_len + idx  # flat index into a (n, chars_len) row-major array

        vec_frq = np.zeros((n, chars_len), dtype=np.float32)
        vec_pre = np.zeros((n, chars_len), dtype=np.float32)
        vec_suc = np.zeros((n, chars_len), dtype=np.float32)

        np.add.at(vec_frq.reshape(-1), flat, (1.0 / wl).astype(np.float32))
        np.add.at(vec_pre.reshape(-1), flat,
                (i_valid * (i_valid + 1) / (2 * wl ** 2)).astype(np.float32))
        np.add.at(vec_suc.reshape(-1), flat,
                ((wl - 1 - i_valid) * (wl - i_valid) / (2 * wl ** 2)).astype(np.float32))

        # phase blocks: each band's cos/sin built as its own contiguous array,
        # then concatenated (avoids the non-contiguous-slice pitfall)
        phase_parts = []
        for freq in PHASE_FREQS:
            theta = freq * np.pi * pos
            cos_arr = np.zeros((n, chars_len), dtype=np.float32)
            sin_arr = np.zeros((n, chars_len), dtype=np.float32)
            np.add.at(cos_arr.reshape(-1), flat, (np.cos(theta) / wl).astype(np.float32))
            np.add.at(sin_arr.reshape(-1), flat, (np.sin(theta) / wl).astype(np.float32))
            phase_parts.append(cos_arr)
            phase_parts.append(sin_arr)

        vec_phase = np.concatenate(phase_parts, axis=1)

        idx_i = idx_all[:, :-1]
        idx_j = idx_all[:, 1:]
        valid_pair = (idx_i >= 0) & (idx_j >= 0)

        word_id2, i2 = np.nonzero(valid_pair)
        ii = idx_i[word_id2, i2]
        jj = idx_j[word_id2, i2]
        wl2 = lens[word_id2].astype(np.float32)

        pair_id = ii * chars_len + jj
        bucket = pair_id % ADJ_DIM

        flat_adj = word_id2 * ADJ_DIM + bucket

        vec_adj = np.zeros((n, ADJ_DIM), dtype=np.float32)
        np.add.at(vec_adj.reshape(-1), flat_adj, (1.0 / wl2).astype(np.float32))

        return np.concatenate([vec_frq, vec_pre, vec_suc, vec_phase, vec_adj], axis=1)
    
    def build(self, entries: list[str]):
        """
        Build the FAISS index using the provided entries.
        
        Args:
            entries (list[str]): A list of strings to vectorize and index.
        """
        self.entries = entries
        self.vectors = self.vectorize_batch(entries)
        self._build_index()
        
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
            
        return self

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
    
    def load_or_build(self, filepath: str, entries: list[str]):
        """
        Load the FAISS index from a file if it exists; otherwise, build the index from the provided entries and save it.
        
        Args:
            filepath (str): The path to the file where the index should be loaded from or saved to.
            entries (list[str]): A list of strings to vectorize and index if the index file does not exist.
        """
        if os.path.exists(filepath):
            return self.load(filepath)
        else:
            return self.build(entries).save(filepath)

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
        
        faiss.omp_set_num_threads(self.num_threads) # Ensure the number of threads is set before performing the search
        
        query_vectors = self.vectorize_batch(queries)
        distances, labels = self.index.search(query_vectors, k)
        
        results = []
        for query, idx, dists in zip(queries, labels, distances):
            result = [(self.entries[idx], dist) for idx, dist in zip(idx, dists) if idx != -1]
            results.append((query, result))
        
        return results
    
    def _build_index(self):
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
        
        faiss.omp_set_num_threads(self.num_threads) # Ensure the number of threads is set before building the index
        
        dim = self.vectors.shape[1]
        
        index = faiss.index_factory(dim, f"HNSW{self.M}", self.metric)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef
        index.add(self.vectors)
        
        self.index = index
        return index
