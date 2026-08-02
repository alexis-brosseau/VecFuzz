from vecfuzz import VecFuzz

# Build a small list of words
words = ["apple", "banana", "orange", "peach", "pineapple"]

# Create the VecFuzz instance and build the index
vecfuzz = VecFuzz().build(words)

# Look up 3 nearest neighbours for each fuzzy query
queries = ["aple", "bannana", "orng"]
results = vecfuzz.lookup(queries, k=3)

for query, candidates in results:
    print(f"Candidates for '{query}':")
    for candidate, distance in candidates:
        print(f"  -> {candidate} (L1 distance: {distance:.4f})")