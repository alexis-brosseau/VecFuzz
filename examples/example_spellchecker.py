from vecfuzz import VecFuzz
from spellchecker import SpellChecker

if __name__ == "__main__":
    # Get words from pyspellchecker's frequency dictionary
    words = list(SpellChecker().word_frequency.dictionary.keys())
    typos = ["teh", "recieve", "definately", "occured", "publically"]
    
    print(f"Building vector index for {len(words)} words...")
    index = VecFuzz().index(words)
    
    print(f"Looking up candidates...\n")
    results = index.lookup(typos, k=3)

    for query, candidates in results:
        print(f"Candidates for '{query}':")
        for candidate, distance in candidates:
            print(f"  → {candidate} (L1 distance: {distance:.4f})")