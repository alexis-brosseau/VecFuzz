from typing import Callable
from dataclasses import dataclass
from functools import partial
import pickle
import io
import zipfile
import numpy as np
import faiss

@dataclass
class VecContext:
    """
    A neutral, pre-computed representation of words.
    Contains no vector-specific logic; purely represents the parsed characters.
    """
    # --- Core Metadata ---
    words: list[str]                  # The original, cleaned strings
    num_chars: int                    # Total size of the character vocabulary
    
    # --- 1D Word-Level Properties ---
    word_lengths: np.ndarray          # Shape: (batch_size,). The actual length of each word.
    
    # --- 2D Grid (For sequence-aware operations like bigrams) ---
    char_matrix: np.ndarray           # Shape: (batch_size, max_len). Char IDs, -1 for pad/unknown.
    
    # --- 1D Flattened Arrays (For fast np.add.at on valid characters only) ---
    word_ids: np.ndarray              # Shape: (num_valid_chars,). Which word each valid char belongs to.
    char_ids: np.ndarray              # Shape: (num_valid_chars,). The vocabulary ID of each valid char.
    char_positions: np.ndarray        # Shape: (num_valid_chars,). 0-based position of the char in its word.
    expanded_word_lengths: np.ndarray # Shape: (num_valid_chars,). The word length, repeated for each valid char.
    flat_char_indices: np.ndarray     # Shape: (num_valid_chars,). Pre-computed (word_id * num_chars + char_id).


class VecFuzz:
    """
    VecFuzz provides functionality for vectorizing strings and performing fuzzy matching 
    using FAISS HNSW indexing. Vectorization is handled by modular, standalone functions.
    """
    
    def __init__(
        self, 
        vectorizers: list[Callable[[VecContext], np.ndarray]],
        metric: int = faiss.METRIC_L1,
        chars: str = "aàbcçdeéèêëfghiïjklmnñoöpqrstuvwxyz0123456789-̧ '. ", 
        ef_construction: int = 200, 
        M: int = 32, 
        ef: int = 64, 
        num_threads: int | None = None
    ):
        """
        Initialize VecFuzz. The vectorization schema is frozen at initialization.
        
        Args:
            chars (str): A string containing all valid characters for vectorization.
            vectorizers (list[Callable]): List of pure functions taking VecContext -> np.ndarray.
            ef_construction (int): The depth of the search during index construction for FAISS HNSW.
            M (int): The number of bi-directional links created for every new element.
            ef (int): The depth of the search for FAISS HNSW.
            num_threads (int): Number of threads used to build/search the index.
        """
        self._chars = chars
        self._chars_len = len(chars)
        self._char_idx = {c: i for i, c in enumerate(chars)}
        
        self.metric = metric
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self.num_threads = num_threads or faiss.omp_get_max_threads() or 1
        
        # Prevent any further modifications to the vectorizers list after initialization
        self._vectorizers = tuple(vectorizers)
        
        self.entries = None
        self.vectors = None
        self.index = None

    def vectorize(self, words: list[str]) -> np.ndarray:
        """
        Vectorize a list of words using the frozen list of callables.
        
        Args:
            words (list[str]): A list of strings to vectorize.
            
        Returns:
            np.ndarray: A 2D numpy array of shape len(words).
        """
        ctx = self._build_context(words)
        parts = [func(ctx) for func in self._vectorizers]
        return np.concatenate(parts, axis=1)

    def build(self, entries: list[str]):
        """Build the FAISS index using the provided entries."""
        self.entries = entries
        self.vectors = self.vectorize(entries)
        self._build_index()
        return self

    def save(self, filepath: str = "index.zip"):
        """Save the vector representations and the FAISS index to a file."""
        faiss_buffer = io.BytesIO()
        writer = faiss.PyCallbackIOWriter(faiss_buffer.write)
        faiss.write_index(self.index, writer)

        metadata_buffer = io.BytesIO()
        pickle.dump({'entries': self.entries, 'vectors': self.vectors}, metadata_buffer)

        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('metadata.pkl', metadata_buffer.getvalue())
            zf.writestr('faiss.index', faiss_buffer.getvalue())
        return self

    def load(self, filepath: str = "index.zip"):
        """Load the vector representations from a file and reconstruct the FAISS index."""
        with zipfile.ZipFile(filepath, 'r') as zf:
            metadata_bytes = zf.read('metadata.pkl')
            data = pickle.loads(metadata_bytes)
            self.entries = data['entries']
            self.vectors = data['vectors']

            faiss_bytes = zf.read('faiss.index')
            reader = faiss.PyCallbackIOReader(io.BytesIO(faiss_bytes).read)
            self.index = faiss.read_index(reader)
        return self

    def lookup(self, queries: list[str], k: int = 1):
        """Perform a similarity search on the index for a given set of queries."""
        if self.index is None:
            raise ValueError("The index has not been built yet. Please call the `build` method before performing lookups.")
            
        faiss.omp_set_num_threads(self.num_threads)
        query_vectors = self.vectorize(queries)
        distances, labels = self.index.search(query_vectors, k)
        
        results = []
        for query, idx, dists in zip(queries, labels, distances):
            result = [(self.entries[i], dist) for i, dist in zip(idx, dists) if i != -1]
            results.append((query, result))
        return results

    def _build_index(self):
        """Construct the FAISS HNSW Index based on the built corpus vectors."""
        faiss.omp_set_num_threads(self.num_threads)
        dim = self.vectors.shape[1]
            
        index = faiss.index_factory(dim, f"HNSW{self.M}", self.metric)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef
        index.add(self.vectors)
        self.index = index
        return index
    
    def _build_context(self, words: list[str]) -> VecContext:
        """Parses strings into the shared, neutral VecContext. Happens exactly once per batch."""
        words = [w.strip().lower() for w in words]
        n = len(words)
        
        if n == 0:
            return VecContext(
                words=[], num_chars=self._chars_len,
                word_lengths=np.array([], dtype=np.int64),
                char_matrix=np.empty((0, 0), dtype=np.int64),
                word_ids=np.array([], dtype=np.int64),
                char_ids=np.array([], dtype=np.int64),
                char_positions=np.array([], dtype=np.int64),
                expanded_word_lengths=np.array([], dtype=np.float32),
                flat_char_indices=np.array([], dtype=np.int64)
            )

        lengths = np.array([len(w) for w in words], dtype=np.int64)
        max_len = int(lengths.max())
        
        # 1. Build the neutral 2D char_matrix
        char_matrix = np.full((n, max_len), -1, dtype=np.int64)
        for r, w in enumerate(words):
            for c, ch in enumerate(w):
                char_matrix[r, c] = self._char_idx.get(ch, -1)
                
        # 2. Extract 1D flattened arrays for valid characters only
        i = np.arange(max_len)[None, :]
        mask = (i < lengths[:, None]) & (char_matrix >= 0)
        word_ids, char_positions = np.nonzero(mask)
        
        char_ids = char_matrix[word_ids, char_positions]
        exp_lengths = lengths[word_ids].astype(np.float32)
        flat_indices = word_ids * self._chars_len + char_ids

        return VecContext(
            words=words, 
            num_chars=self._chars_len,
            word_lengths=lengths,
            char_matrix=char_matrix,
            word_ids=word_ids, 
            char_ids=char_ids, 
            char_positions=char_positions,
            expanded_word_lengths=exp_lengths, 
            flat_char_indices=flat_indices
        )


class VectorizerOp:
    """
    A callable wrapper that allows vectorizer functions to be multiplied 
    or divided by a scalar weight (e.g., `vectorizer * 2.0` or `vectorizer / 3.0`).
    """
    
    def __init__(self, func):
        self._func = func
        self.__name__ = getattr(func, '__name__', 'vectorizer')
        self.__doc__ = getattr(func, '__doc__', '')

    def __call__(self, ctx, *args, **kwargs):
        return self._func(ctx, *args, **kwargs)

    def __mul__(self, scalar):
        def scaled(ctx, *args, **kwargs):
            return self._func(ctx, *args, **kwargs) * scalar
        op = VectorizerOp(scaled)
        op.__name__ = f"{self.__name__}_x{scalar}"
        return op

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        if scalar == 0: raise ZeroDivisionError("Cannot divide vectorizer by zero")
        def scaled(ctx, *args, **kwargs):
            return self._func(ctx, *args, **kwargs) / scalar
        op = VectorizerOp(scaled)
        op.__name__ = f"{self.__name__}_/{scalar}"
        return op

    def __rtruediv__(self, scalar):
        if scalar == 0:
            def zero_func(ctx, *args, **kwargs):
                return np.zeros_like(self._func(ctx, *args, **kwargs))
            return VectorizerOp(zero_func)
        def scaled(ctx, *args, **kwargs):
            return scalar / self._func(ctx, *args, **kwargs)
        op = VectorizerOp(scaled)
        op.__name__ = f"{scalar}/{self.__name__}"
        return op

    def __pow__(self, exponent):
        """Handles `vectorizer ** exponent`"""
        def powered(ctx, *args, **kwargs):
            # Use np.power to handle negative bases with fractional exponents safely if needed,
            # but standard ** is fine for positive vectors.
            return np.power(self._func(ctx, *args, **kwargs), exponent)
        
        op = VectorizerOp(powered)
        op.__name__ = f"{self.__name__}^{exponent}"
        return op
    
    def norm(self, ord: int) -> "VectorizerOp":
        """
        Returns a new VectorizerOp that normalizes the output vector 
        to have a unit norm (L2 by default) per row (per word).
        
        Args:
            ord (int): The order of the norm. 2 for Euclidean (L2), 1 for Manhattan (L1).
        """
        def normed(ctx, *args, **kwargs):
            vec = self._func(ctx, *args, **kwargs)
            
            # Calculate norm per row (each word's vector gets normalized independently)
            norms = np.linalg.norm(vec, ord=ord, axis=1, keepdims=True)
            
            # Prevent division by zero for empty words or zero-vectors
            norms = np.where(norms == 0, 1.0, norms)
            
            return vec / norms
        
        op = VectorizerOp(normed)
        op.__name__ = f"norm{ord}({self.__name__})"
        return op
    
    def params(self, *args, **kwargs) -> "VectorizerOp":
        """
        EXPLICIT CONFIGURATION MODE.
        Binds parameters to this vectorizer and returns a new VectorizerOp.
        """
        # Using functools.partial is highly optimized in C
        bound_func = partial(self._func, *args, **kwargs)
        
        new_op = VectorizerOp(bound_func)
        new_op.__doc__ = self.__doc__
        
        # Generate a clean, readable name for debugging/introspection
        param_strs = [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        if param_strs:
            new_op.__name__ = f"{self.__name__}({', '.join(param_strs)})"
        else:
            new_op.__name__ = self.__name__
            
        return new_op
    
    
def vectorizer(func):
    """Decorator to convert a raw vectorizer function into a VectorizerOp."""
    return VectorizerOp(func)


class Vectorizer:

    @staticmethod
    @vectorizer
    def frequency(ctx: VecContext) -> np.ndarray:
        """Character frequency, normalized by word length."""
        n = ctx.word_lengths.shape[0]
        vec = np.zeros((n, ctx.num_chars), dtype=np.float32)
        np.add.at(vec.reshape(-1), ctx.flat_char_indices, (1.0 / ctx.expanded_word_lengths).astype(np.float32))
        return vec

    @staticmethod
    @vectorizer
    def density_backward(ctx: VecContext) -> np.ndarray:
        """Preceding-position density: sum of positions that came before each character."""
        n = ctx.word_lengths.shape[0]
        vec = np.zeros((n, ctx.num_chars), dtype=np.float32)
        pos = ctx.char_positions
        wl = ctx.expanded_word_lengths
        np.add.at(vec.reshape(-1), ctx.flat_char_indices, (pos * (pos + 1) / (wl ** 2)).astype(np.float32))
        return vec

    @staticmethod
    @vectorizer
    def density_forward(ctx: VecContext) -> np.ndarray:
        """Succeeding-position density: sum of positions that came after each character."""
        n = ctx.word_lengths.shape[0]
        vec = np.zeros((n, ctx.num_chars), dtype=np.float32)
        pos = ctx.char_positions
        wl = ctx.expanded_word_lengths
        np.add.at(vec.reshape(-1), ctx.flat_char_indices, ((wl - 1 - pos) * (wl - pos) / (wl ** 2)).astype(np.float32))
        return vec
    
    @staticmethod
    @vectorizer
    def density(ctx: VecContext) -> np.ndarray:
        """
            Concatenation of backward and forward density vectors.
        """
        
        backward = Vectorizer.density_backward(ctx)
        forward = Vectorizer.density_forward(ctx)
        return np.concatenate([backward, forward], axis=1)
    
    @staticmethod
    @vectorizer
    def position_avg(ctx: VecContext) -> np.ndarray:
        """
        Average position of each character, normalized by word length.
        Positions are 1-based (idx + 1) to avoid zeros.
        """
        n = ctx.word_lengths.shape[0]
        vec = np.zeros((n, ctx.num_chars), dtype=np.float32)
        
        pos_1_based = ctx.char_positions + 1
        np.add.at(vec.reshape(-1), ctx.flat_char_indices, pos_1_based.astype(np.float32))

        counts = np.zeros((n, ctx.num_chars), dtype=np.float32)
        np.add.at(counts.reshape(-1), ctx.flat_char_indices, 1.0)
        
        mask = counts > 0
        vec[mask] /= counts[mask]
        vec /= ctx.word_lengths[:, None]
        
        return vec

    @staticmethod
    @vectorizer
    def position_phase(ctx: VecContext, freqs: tuple[float, ...] = (1.0, 2.0)) -> np.ndarray:
        """
        Phase-encoded position: sinusoidal expansion of each character's position.
        
        Args:
            freqs (tuple[float]): Frequencies for the sinusoidal encoding.
        """
        n = ctx.word_lengths.shape[0]
        pos = (ctx.char_positions + 1) / ctx.expanded_word_lengths
        phase_parts = []
        
        for freq in freqs:
            theta = freq * np.pi * pos
            cos_arr = np.zeros((n, ctx.num_chars), dtype=np.float32)
            sin_arr = np.zeros((n, ctx.num_chars), dtype=np.float32)
            
            np.add.at(cos_arr.reshape(-1), ctx.flat_char_indices, (np.cos(theta) / ctx.expanded_word_lengths).astype(np.float32))
            np.add.at(sin_arr.reshape(-1), ctx.flat_char_indices, (np.sin(theta) / ctx.expanded_word_lengths).astype(np.float32))
            
            phase_parts.extend([cos_arr, sin_arr])
            
        return np.concatenate(phase_parts, axis=1)
    
    @staticmethod
    @vectorizer
    def position_rbf(ctx: VecContext, centers: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0), sigma: float = 0.15) -> np.ndarray:
        """
        Radial Basis Function (Gaussian) positional encoding.
        
        It uses a Gaussian kernel, which is Smooth AND Local. 
        A deletion only shifts characters within the local "window" of the Gaussian. 
        Dimensions far from the deletion remain mathematically untouched, keeping the 
        L1 distance tiny while maintaining continuous positional awareness.
        
        Args:
            centers (tuple[float]): The relative positions (0.0 to 1.0) to place Gaussian centers.
            sigma (float): The width of the Gaussian. Controls how much overlap there is between centers.
        """
        n = ctx.word_lengths.shape[0]
        num_centers = len(centers)
        
        if n == 0:
            return np.zeros((0, ctx.num_chars * num_centers), dtype=np.float32)

        vec = np.zeros((n, ctx.num_chars * num_centers), dtype=np.float32)
        
        # Relative positions (0.0 to 1.0)
        rel_pos = ctx.char_positions.astype(np.float32) / ctx.expanded_word_lengths
        
        # Precompute the denominator for the Gaussian
        denom = 2.0 * (sigma ** 2)
        
        for c_idx, center in enumerate(centers):
            # Calculate Gaussian activation for this specific center
            # act shape: (num_valid_chars,)
            act = np.exp(-((rel_pos - center) ** 2) / denom)
            
            # Target flat index: word_id * (chars * centers) + char_id * centers + center_idx
            target = ctx.word_ids * (ctx.num_chars * num_centers) + ctx.char_ids * num_centers + c_idx
            
            # Accumulate
            np.add.at(vec.reshape(-1), target, act.astype(np.float32))
            
        # Normalize by word length so longer words don't dominate the magnitude
        vec /= ctx.word_lengths[:, None]
        
        return vec

    @staticmethod
    @vectorizer
    def bigram(ctx: VecContext, dim: int = 192) -> np.ndarray:
        """
        Bigram frequency vectorization. Each bigram is hashed into a fixed number of buckets.
        
        Args:
            dim (int): The dimensionality of the output vector. Bigrams are hashed into this many buckets.
        """
        n = ctx.word_lengths.shape[0]
        
        # Derives bigrams purely from the neutral 2D char_matrix
        idx_i = ctx.char_matrix[:, :-1]
        idx_j = ctx.char_matrix[:, 1:]
        
        valid_pair = (idx_i >= 0) & (idx_j >= 0)
        word_ids, col_ids = np.nonzero(valid_pair)
        
        ii = idx_i[word_ids, col_ids]
        jj = idx_j[word_ids, col_ids]
        
        wl = ctx.word_lengths[word_ids].astype(np.float32)
        pair_id = ii * ctx.num_chars + jj
        bucket = pair_id % dim
        
        flat_adj = word_ids * dim + bucket
        vec = np.zeros((n, dim), dtype=np.float32)
        np.add.at(vec.reshape(-1), flat_adj, (1.0 / wl).astype(np.float32))
        return vec