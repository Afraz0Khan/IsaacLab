"""
Token analysis utilities for STARC v2

Demonstrates token savings from compact matrix format
"""

def estimate_token_savings(n_rewards: int) -> dict:
    """
    Estimate token savings from using compact matrix format
    
    Args:
        n_rewards: Number of rewards in the matrix
        
    Returns:
        Dict with token estimates and savings
    """
    
    # Rough token estimates (1 token ≈ 4 characters for numbers/symbols)
    chars_per_distance = 8  # "1.2345, " format
    tokens_per_distance = chars_per_distance / 4
    
    # Full matrix: n x n entries
    full_matrix_distances = n_rewards * n_rewards
    full_matrix_tokens = full_matrix_distances * tokens_per_distance
    
    # Compact matrix: upper triangle only
    # Number of unique pairs = n*(n-1)/2 (excluding diagonal)
    compact_matrix_distances = n_rewards * (n_rewards - 1) // 2
    compact_matrix_tokens = compact_matrix_distances * tokens_per_distance
    
    # Additional overhead (headers, formatting, etc.)
    overhead_tokens = 100  # Rough estimate
    
    full_total = full_matrix_tokens + overhead_tokens
    compact_total = compact_matrix_tokens + overhead_tokens
    
    savings = full_total - compact_total
    savings_percent = (savings / full_total) * 100
    
    return {
        "n_rewards": n_rewards,
        "full_matrix": {
            "distances": full_matrix_distances,
            "estimated_tokens": int(full_total)
        },
        "compact_matrix": {
            "distances": compact_matrix_distances,
            "estimated_tokens": int(compact_total)
        },
        "savings": {
            "tokens_saved": int(savings),
            "percent_saved": round(savings_percent, 1)
        }
    }

def print_token_analysis():
    """Print token analysis for different matrix sizes"""
    print("📊 Token Savings Analysis: Compact vs Full Matrix Format")
    print("=" * 60)
    print(f"{'Rewards':<8} {'Full Tokens':<12} {'Compact Tokens':<15} {'Saved':<8} {'% Saved':<8}")
    print("-" * 60)
    
    for n in [5, 10, 16, 20, 30]:
        analysis = estimate_token_savings(n)
        full = analysis["full_matrix"]["estimated_tokens"]
        compact = analysis["compact_matrix"]["estimated_tokens"]
        saved = analysis["savings"]["tokens_saved"]
        percent = analysis["savings"]["percent_saved"]
        
        print(f"{n:<8} {full:<12} {compact:<15} {saved:<8} {percent:<8}%")
    
    print("\n💡 Key Insights:")
    print("- Token savings increase with matrix size")
    print("- For 16 rewards (typical iteration), saves ~40-45% tokens")
    print("- Larger matrices see even greater savings")
    print("- No information loss - all unique distances preserved")

if __name__ == "__main__":
    print_token_analysis() 