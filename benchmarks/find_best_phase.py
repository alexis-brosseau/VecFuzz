import numpy as np
import faiss
import itertools
from vecfuzz import VecFuzz, Vectorizer
from benchmark import load_vocabulary, generate_error_cases, TYPO_TYPES, DEFAULT_EDIT_LEVELS

def find_best_phase_params():
    print("Loading vocabulary...")
    vocab = load_vocabulary(max_words=50000, seed=0)
    
    # Generate a representative subset of cases for fast evaluation
    cases_per_combo = 200
    print("Generating error cases...")
    all_cases = generate_error_cases(vocab, cases_per_combo, DEFAULT_EDIT_LEVELS, seed=42)
    
    # Group cases by error type
    cases_by_type = {typo: [] for typo in TYPO_TYPES}
    for case in all_cases:
        cases_by_type[case['error_type']].append(case)
        
    queries_by_type = {typo: [c['query'] for c in cases] for typo, cases in cases_by_type.items()}
    targets_by_type = {typo: [c['target'] for c in cases] for typo, cases in cases_by_type.items()}

    # --- THE SEARCH SPACE ---
    # We use a pool of mathematically significant harmonics.
    # 0.5: Sub-harmonic (captures very long-range global shape)
    # 1.0: Fundamental frequency
    # 2.0 - 5.0: Higher harmonics (capture local positional details for substitutions)
    CANDIDATE_FREQS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    MAX_FREQ_COUNT = 3
    
    # Generate all unique combinations (order doesn't matter for concatenated sine waves)
    search_space = []
    for num_freqs in range(1, MAX_FREQ_COUNT + 1):
        for combo in itertools.combinations_with_replacement(CANDIDATE_FREQS, num_freqs):
            search_space.append(combo)
            
    print(f"Testing {len(search_space)} unique harmonic combinations...")
    
    # Trackers for individual error types
    best_configs = {
        typo: {'score': -1.0, 'freqs': None} 
        for typo in TYPO_TYPES
    }
    
    # Tracker for the overall average
    best_overall = {
        'score': -1.0, 
        'freqs': None,
        'breakdown': {}
    }
    
    total_combos = len(search_space)
    
    for idx, freqs in enumerate(search_space, 1):
        # We use L2 + Norm(2) (Cosine Similarity) because it is the undisputed king 
        # for Phase/RFB vectors, neutralizing the L1 deletion penalty while keeping substitution sharp.
        vf = VecFuzz(
            metric=faiss.METRIC_L2,
            vectorizers=[
                Vectorizer.position_phase.params(freqs=freqs).norm(2)
            ]
        )
        vf.build(vocab)
        
        recalls = {}
        for typo in TYPO_TYPES:
            queries = queries_by_type[typo]
            targets = targets_by_type[typo]
            
            results = vf.lookup(queries, k=1)
            hits = sum(1 for res, target in zip(results, targets) if res[1] and res[1][0][0] == target)
            recall = hits / len(targets)
            recalls[typo] = recall
            
            # Update individual bests
            if recall > best_configs[typo]['score']:
                best_configs[typo]['score'] = recall
                best_configs[typo]['freqs'] = freqs

        # Calculate the average across all 4 error types
        avg_recall = sum(recalls.values()) / len(TYPO_TYPES)
        
        is_overall_best = False
        if avg_recall > best_overall['score']:
            best_overall['score'] = avg_recall
            best_overall['freqs'] = freqs
            best_overall['breakdown'] = recalls.copy()
            is_overall_best = True

        # Print progress (every 10 iterations to keep console clean)
        print(f"[{idx}/{total_combos}] freqs={freqs} | "
                f"Sub:{recalls['substitution']:.3f} Trans:{recalls['transposition']:.3f} "
                f"Ins:{recalls['insertion']:.3f} Del:{recalls['deletion']:.3f} | "
                f"AVG:{avg_recall:.4f} {'<-- OVERALL BEST' if is_overall_best else ''}")

    # --- OUTPUT RESULTS ---
    print("\n" + "="*80)
    print("BEST OVERALL CONFIGURATION (Pareto Optimal Average)")
    print("="*80)
    print(f"  Best Frequencies: {best_overall['freqs']}")
    print(f"  Average Recall:   {best_overall['score']:.4f}")
    print(f"  Breakdown:")
    for typo, score in best_overall['breakdown'].items():
        print(f"    {typo.capitalize():15s}: {score:.4f}")
    
    freqs_str = ", ".join([f"{f:.1f}" if f % 1 == 0 else f"{f:.2f}" for f in best_overall['freqs']])
    print(f"\n  Python Config:")
    print(f"    VecFuzz(")
    print(f"        metric=faiss.METRIC_L2,")
    print(f"        vectorizers=[")
    print(f"            Vectorizer.phase_position.params(freqs=({freqs_str})).norm(2)")
    print(f"        ]")
    print(f"    )")

    print("\n" + "="*80)
    print("BEST CONFIGURATIONS PER ERROR TYPE (For Custom Pareto Weighting)")
    print("="*80)
    
    for typo in TYPO_TYPES:
        cfg = best_configs[typo]
        print(f"\n{typo.upper()}")
        print(f"  Best Frequencies: {cfg['freqs']} | Recall: {cfg['score']:.4f}")
        freqs_str = ", ".join([f"{f:.1f}" if f % 1 == 0 else f"{f:.2f}" for f in cfg['freqs']])
        print(f"    freqs=({freqs_str})")

if __name__ == "__main__":
    find_best_phase_params()