import numpy as np
import faiss
import itertools
from vecfuzz import VecFuzz, Vectorizer
from benchmark import load_vocabulary, generate_error_cases, TYPO_TYPES, DEFAULT_EDIT_LEVELS

def find_best_rbf_params():
    print("Loading vocabulary...")
    vocab = load_vocabulary(max_words=50000, seed=0)
    
    # Generate a representative subset of cases for faster grid search
    cases_per_combo = 200
    print("Generating error cases...")
    all_cases = generate_error_cases(vocab, cases_per_combo, DEFAULT_EDIT_LEVELS, seed=42)
    
    # Group cases by error type
    cases_by_type = {typo: [] for typo in TYPO_TYPES}
    for case in all_cases:
        cases_by_type[case['error_type']].append(case)
        
    queries_by_type = {typo: [c['query'] for c in cases] for typo, cases in cases_by_type.items()}
    targets_by_type = {typo: [c['target'] for c in cases] for typo, cases in cases_by_type.items()}

    # Grid search parameters
    c_values = [0, 5, 10, 20] 
    s_values = [0.01, 0.05, 0.10, 0.15, 0.18, 0.25, 0.35]
    
    # Trackers for individual error types
    best_configs = {
        typo: {'score': -1.0, 'c': None, 's': None, 'centers': None} 
        for typo in TYPO_TYPES
    }
    
    # NEW: Tracker for the overall average across all error types
    best_overall = {
        'score': -1.0, 
        'c': None, 
        's': None, 
        'centers': None,
        'breakdown': {}
    }
    
    total_combos = len(c_values) * len(s_values)
    print(f"Starting grid search over {total_combos} combinations...")
    
    for idx, (c, s) in enumerate(itertools.product(c_values, s_values), 1):
        centers = tuple(np.linspace(0.0, 1.0, c).tolist())
        
        vf = VecFuzz(
            metric=faiss.METRIC_L2,
            vectorizers=[
                Vectorizer.position_rbf.params(centers=centers, sigma=s).norm(2)
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
                best_configs[typo]['c'] = c
                best_configs[typo]['s'] = s
                best_configs[typo]['centers'] = centers

        # Calculate the average across all 4 error types
        avg_recall = sum(recalls.values()) / len(TYPO_TYPES)
        
        is_overall_best = False
        if avg_recall > best_overall['score']:
            best_overall['score'] = avg_recall
            best_overall['c'] = c
            best_overall['s'] = s
            best_overall['centers'] = centers
            best_overall['breakdown'] = recalls.copy()
            is_overall_best = True

        # Print progress
        print(f"[{idx}/{total_combos}] c={c}, s={s:.3f} | "
              f"Sub:{recalls['substitution']:.3f} Trans:{recalls['transposition']:.3f} "
              f"Ins:{recalls['insertion']:.3f} Del:{recalls['deletion']:.3f} | "
              f"AVG:{avg_recall:.4f} {'<-- OVERALL BEST' if is_overall_best else ''}")

    print("\n" + "="*80)
    print("BEST OVERALL CONFIGURATION (Pareto Optimal Average)")
    print("="*80)
    print(f"  Best centers (c): {best_overall['c']}")
    print(f"  Best sigma (s):         {best_overall['s']}")
    print(f"  Average Recall:         {best_overall['score']:.4f}")
    print(f"  Breakdown:")
    for typo, score in best_overall['breakdown'].items():
        print(f"    {typo.capitalize():15s}: {score:.4f}")
    
    centers_str = ", ".join([f"{x:.4f}" if x % 1 != 0 else f"{x:.1f}" for x in best_overall['centers']])
    print(f"\n  Python Config:")
    print(f"    VecFuzz(")
    print(f"        metric=faiss.METRIC_L2,")
    print(f"        vectorizers=[")
    print(f"            Vectorizer.position_rbf.params(centers=({centers_str}), sigma={best_overall['s']}).norm(2)")
    print(f"        ]")
    print(f"    )")

    print("\n" + "="*80)
    print("BEST CONFIGURATIONS PER ERROR TYPE (For Custom Pareto Weighting)")
    print("="*80)
    
    for typo in TYPO_TYPES:
        cfg = best_configs[typo]
        print(f"\n{typo.upper()}")
        print(f"  Best c: {cfg['c']} | Best s: {cfg['s']} | Recall: {cfg['score']:.4f}")
        centers_str = ", ".join([f"{x:.4f}" if x % 1 != 0 else f"{x:.1f}" for x in cfg['centers']])
        print(f"    centers=({centers_str}), sigma={cfg['s']}")

if __name__ == "__main__":
    find_best_rbf_params()