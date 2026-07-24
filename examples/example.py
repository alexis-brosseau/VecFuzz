from vecfuzz import VecFuzz

# Build a small list of words
words = ["apple", "banana", "orange", "peach", "pineapple"]

# Create the index
index = VecFuzz().build_index(words)

# Look up 3 nearest neighbours for each fuzzy query
queries = ["aple", "bannana", "orng"]
results = index.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for candidate, distance in candidates:
        print(f"  → {candidate} (L1 distance: {distance:.4f})")