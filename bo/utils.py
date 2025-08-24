#!/usr/bin/env python3
"""
Utility functions for Bayesian optimization of HalfCheetah reward functions.

This module provides configuration management, analysis tools, and helper functions
for the BO setup using canonical sum-of-features reward functions.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns

# Set matplotlib style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class BOConfig:
    """Configuration class for Bayesian optimization settings."""
    
    # Default settings for different use cases
    FAST_TEST = {
        'train_timesteps': 10000,
        'eval_episodes': 3,
        'eval_timesteps': 2000,
        'n_iterations': 5,
        'n_initial': 3
    }
    
    STANDARD = {
        'train_timesteps': 80000,
        'eval_episodes': 10,
        'eval_timesteps': 2000, 
        'n_iterations': 20,
        'n_initial': 5
    }
    
    THOROUGH = {
        'train_timesteps': 150000,
        'eval_episodes': 20,
        'eval_timesteps': 2000,
        'n_iterations': 50,
        'n_initial': 10
    }
    
    def __init__(self, preset: str = "standard", **kwargs):
        """
        Initialize configuration.
        
        Args:
            preset: Configuration preset ("fast_test", "standard", or "thorough")
            **kwargs: Override specific settings
        """
        if preset.lower() == "fast_test":
            base_config = self.FAST_TEST.copy()
        elif preset.lower() == "standard":
            base_config = self.STANDARD.copy()
        elif preset.lower() == "thorough":
            base_config = self.THOROUGH.copy()
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Override with any provided kwargs
        base_config.update(kwargs)
        
        # Set attributes
        for key, value in base_config.items():
            setattr(self, key, value)
        
        # Additional settings
        self.acquisition_function = kwargs.get('acquisition_function', 'UCB')
        self.kappa = kwargs.get('kappa', 2.0)
        self.seed = kwargs.get('seed', None)
        self.verbose = kwargs.get('verbose', True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
            and attr not in ['FAST_TEST', 'STANDARD', 'THOROUGH']
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BOConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create instance with dummy preset and override all values
        instance = cls("standard")
        for key, value in config_dict.items():
            setattr(instance, key, value)
        
        return instance


class BOAnalyzer:
    """Analysis tools for Bayesian optimization results."""
    
    def __init__(self, results_file: Union[str, Path]):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to BO results JSON file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.df = pd.DataFrame(self.results['iteration_results'])
        
        # Get feature information if available
        self.feature_analysis = self.results.get('feature_analysis', {})
        self.feature_names = self.feature_analysis.get('feature_names', [])
        if not self.feature_names:
            # Fallback to generic names
            n_features = self.results['settings'].get('n_features', 21)
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
    def plot_convergence(self, save_path: Optional[str] = None, figsize: Tuple = (12, 8)) -> None:
        """Plot detailed convergence analysis."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Bayesian Optimization Convergence Analysis (Canonical Reward)', fontsize=16)
        
        # Plot 1: Objective values over iterations
        ax1 = axes[0, 0]
        iterations = range(1, len(self.results['all_values']) + 1)
        values = self.results['all_values']
        
        ax1.scatter(iterations, values, alpha=0.6, c='blue', s=50, label='Evaluations')
        
        # Running best
        running_best = [max(values[:i+1]) for i in range(len(values))]
        ax1.plot(iterations, running_best, 'r-', linewidth=2, label='Best so far')
        
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Distance Traveled')
        ax1.set_title('Objective Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement over iterations
        ax2 = axes[0, 1]
        improvements = np.diff([0] + running_best)
        ax2.bar(range(1, len(improvements) + 1), improvements, alpha=0.7, color='green')
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Improvement')
        ax2.set_title('Per-Iteration Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Active features over iterations
        ax3 = axes[1, 0]
        if 'active_features' in self.df.columns:
            ax3.plot(self.df.index + 1, self.df['active_features'], 'go-', alpha=0.7)
            ax3.set_xlabel('BO Iteration')
            ax3.set_ylabel('Active Features')
            ax3.set_title('Active Features per Iteration')
        else:
            # Fallback: compute active features from parameters
            params = np.array(self.results['all_parameters'])
            active_features = [np.sum(np.abs(p) > 1e-6) for p in params]
            ax3.plot(iterations, active_features, 'go-', alpha=0.7)
            ax3.set_xlabel('Evaluation')
            ax3.set_ylabel('Active Features')
            ax3.set_title('Active Features per Evaluation')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of objective values
        ax4 = axes[1, 1]
        ax4.hist(values, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(max(values), color='red', linestyle='--', linewidth=2, label=f'Best: {max(values):.3f}')
        ax4.set_xlabel('Distance Traveled')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Evaluations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def parameter_importance(self, figsize: Tuple = (12, 8), save_path: Optional[str] = None,
                           top_n: int = 10) -> Dict:
        """Analyze parameter importance through correlation with objective."""
        params = np.array(self.results['all_parameters'])
        values = np.array(self.results['all_values'])
        
        # Calculate correlations for each feature
        correlations = []
        for i in range(params.shape[1]):
            corr = np.corrcoef(params[:, i], values)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        
        # Create importance analysis
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Importance Analysis', fontsize=16)
        
        # Plot 1: Top positive correlations
        ax1 = axes[0, 0]
        sorted_indices = np.argsort(correlations)[::-1]
        top_positive = sorted_indices[:top_n]
        top_pos_names = [self.feature_names[i][:15] for i in top_positive]  # Truncate names
        top_pos_corrs = [correlations[i] for i in top_positive]
        
        bars = ax1.barh(range(len(top_pos_corrs)), top_pos_corrs, color='green', alpha=0.7)
        ax1.set_yticks(range(len(top_pos_corrs)))
        ax1.set_yticklabels(top_pos_names)
        ax1.set_xlabel('Correlation with Distance')
        ax1.set_title(f'Top {top_n} Positive Features')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top negative correlations
        ax2 = axes[0, 1]
        top_negative = sorted_indices[-top_n:]
        top_neg_names = [self.feature_names[i][:15] for i in top_negative]
        top_neg_corrs = [correlations[i] for i in top_negative]
        
        bars = ax2.barh(range(len(top_neg_corrs)), top_neg_corrs, color='red', alpha=0.7)
        ax2.set_yticks(range(len(top_neg_corrs)))
        ax2.set_yticklabels(top_neg_names)
        ax2.set_xlabel('Correlation with Distance')
        ax2.set_title(f'Top {top_n} Negative Features')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Feature weight distribution in best solution
        ax3 = axes[1, 0]
        best_params = np.array(self.results['best_parameters'])
        active_mask = np.abs(best_params) > 1e-6
        
        if np.any(active_mask):
            active_names = [self.feature_names[i][:15] for i in np.where(active_mask)[0]]
            active_weights = best_params[active_mask]
            
            colors = ['green' if w > 0 else 'red' for w in active_weights]
            bars = ax3.barh(range(len(active_weights)), active_weights, color=colors, alpha=0.7)
            ax3.set_yticks(range(len(active_weights)))
            ax3.set_yticklabels(active_names)
            ax3.set_xlabel('Weight Value')
            ax3.set_title('Active Features in Best Solution')
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No active features', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Active Features in Best Solution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature activity over time
        ax4 = axes[1, 1]
        activity_matrix = np.abs(params) > 1e-6
        activity_counts = np.sum(activity_matrix, axis=0)
        
        sorted_activity = np.argsort(activity_counts)[::-1]
        top_active_names = [self.feature_names[i][:15] for i in sorted_activity[:top_n]]
        top_active_counts = [activity_counts[i] for i in sorted_activity[:top_n]]
        
        bars = ax4.barh(range(len(top_active_counts)), top_active_counts, alpha=0.7, color='orange')
        ax4.set_yticks(range(len(top_active_counts)))
        ax4.set_yticklabels(top_active_names)
        ax4.set_xlabel('Times Active')
        ax4.set_title(f'Most Frequently Active Features')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter importance analysis saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Return importance dictionary
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            importance_dict[name] = {
                'correlation': correlations[i],
                'activity_count': int(activity_counts[i]) if i < len(activity_counts) else 0,
                'best_weight': float(best_params[i]) if i < len(best_params) else 0.0
            }
        
        return importance_dict
    
    def get_summary(self) -> Dict:
        """Get summary statistics of optimization results."""
        values = np.array(self.results['all_values'])
        params = np.array(self.results['all_parameters'])
        
        # Calculate active feature statistics
        active_features_per_eval = [np.sum(np.abs(p) > 1e-6) for p in params]
        best_params = np.array(self.results['best_parameters'])
        active_in_best = np.sum(np.abs(best_params) > 1e-6)
        
        summary = {
            'total_evaluations': len(values),
            'best_value': float(np.max(values)),
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'median_value': float(np.median(values)),
            'improvement_over_initial': float(np.max(values) - values[0]),
            'total_time_hours': self.results.get('total_time_seconds', 0) / 3600,
            'evaluations_per_hour': len(values) / max(1, self.results.get('total_time_seconds', 1) / 3600),
            'best_parameters': self.results['best_parameters'],
            'convergence_rate': self._calculate_convergence_rate(values),
            'total_features': len(self.feature_names),
            'active_features_in_best': int(active_in_best),
            'avg_active_features': float(np.mean(active_features_per_eval)),
            'std_active_features': float(np.std(active_features_per_eval))
        }
        
        return summary
    
    def _calculate_convergence_rate(self, values: np.ndarray) -> float:
        """Calculate convergence rate as improvement in last 25% vs first 25%."""
        n = len(values)
        if n < 8:
            return 0.0
        
        first_quarter = values[:n//4]
        last_quarter = values[3*n//4:]
        
        return float(np.mean(last_quarter) - np.mean(first_quarter))
    
    def print_summary(self) -> None:
        """Print formatted summary of optimization results."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("🔍 BAYESIAN OPTIMIZATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Evaluations: {summary['total_evaluations']}")
        print(f"Best Distance: {summary['best_value']:.3f}")
        print(f"Mean Distance: {summary['mean_value']:.3f} (±{summary['std_value']:.3f})")
        print(f"Median Distance: {summary['median_value']:.3f}")
        print(f"Improvement: +{summary['improvement_over_initial']:.3f}")
        print(f"Total Time: {summary['total_time_hours']:.2f} hours")
        print(f"Evaluation Rate: {summary['evaluations_per_hour']:.1f} evaluations/hour")
        print(f"Convergence Rate: {summary['convergence_rate']:.3f}")
        print(f"\nFeature Statistics:")
        print(f"Total Features: {summary['total_features']}")
        print(f"Active in Best Solution: {summary['active_features_in_best']}")
        print(f"Avg Active per Evaluation: {summary['avg_active_features']:.1f} ± {summary['std_active_features']:.1f}")
        
        # Show top features if available
        if self.feature_analysis and self.feature_analysis.get('most_positive_features'):
            print(f"\nTop Positive Features:")
            for name, corr in self.feature_analysis['most_positive_features'][:3]:
                print(f"  {name}: {corr:.3f}")
        
        print("=" * 60)


def compare_results(result_files: List[Union[str, Path]], 
                   save_path: Optional[str] = None,
                   figsize: Tuple = (12, 8)) -> None:
    """
    Compare results from multiple BO runs.
    
    Args:
        result_files: List of paths to BO result JSON files
        save_path: Path to save comparison plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(result_files)))
    
    for i, (file_path, color) in enumerate(zip(result_files, colors)):
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        values = results['all_values']
        iterations = range(1, len(values) + 1)
        
        # Running best
        running_best = [max(values[:j+1]) for j in range(len(values))]
        
        # Get run info
        n_features = results.get('settings', {}).get('n_features', 'Unknown')
        label = f"Run {i+1} (Best: {max(values):.3f}, {n_features} features)"
        
        plt.plot(iterations, running_best, '-', color=color, linewidth=2, label=label)
        
        # Add scatter of all evaluations with transparency
        plt.scatter(iterations, values, color=color, alpha=0.3, s=20)
    
    plt.xlabel('Evaluation')
    plt.ylabel('Distance Traveled')
    plt.title('Bayesian Optimization Comparison (Canonical Rewards)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_experiment_report(results_file: Union[str, Path], 
                           output_dir: Union[str, Path] = "experiment_report") -> None:
    """
    Create a comprehensive experiment report with all analyses.
    
    Args:
        results_file: Path to BO results JSON file
        output_dir: Directory to save report files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    analyzer = BOAnalyzer(results_file)
    
    print(f"🔬 Creating experiment report in {output_path}")
    
    # Generate all plots
    analyzer.plot_convergence(save_path=output_path / "convergence_analysis.png")
    importance = analyzer.parameter_importance(save_path=output_path / "parameter_importance.png")
    
    # Save summary and feature analysis
    summary = analyzer.get_summary()
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(output_path / "feature_importance.json", 'w') as f:
        json.dump(importance, f, indent=2)
    
    # Create text report
    with open(output_path / "report.txt", 'w') as f:
        f.write("BAYESIAN OPTIMIZATION EXPERIMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results file: {results_file}\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        for key, value in summary.items():
            if key == 'best_parameters':
                f.write(f"Best Parameters: {len(value)} weights\n")
                active_weights = [w for w in value if abs(w) > 1e-6]
                f.write(f"Active Weights: {len(active_weights)}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write(f"\nFEATURE IMPORTANCE:\n")
        f.write("-" * 20 + "\n")
        
        # Top positive features
        sorted_features = sorted(importance.items(), key=lambda x: x[1]['correlation'], reverse=True)
        f.write("Most Positive Features:\n")
        for name, info in sorted_features[:10]:
            f.write(f"  {name}: correlation={info['correlation']:.3f}, "
                   f"activity={info['activity_count']}, "
                   f"best_weight={info['best_weight']:.3f}\n")
    
    print(f"✅ Experiment report saved to {output_path}")
    print(f"   - Convergence analysis: convergence_analysis.png")
    print(f"   - Parameter importance: parameter_importance.png")
    print(f"   - Summary statistics: summary.json")
    print(f"   - Feature importance: feature_importance.json")
    print(f"   - Text report: report.txt") 


# Standalone plotting functions for BayesianOptimizer
def plot_convergence(iterations: List[int], 
                   best_values: List[float],
                   save_path: Optional[str] = None,
                   title: str = "Bayesian Optimization Convergence",
                   use_seed_rewards: bool = False,
                   n_seed_rewards: int = 0) -> None:
    """
    Plot the convergence of Bayesian optimization.
    
    Args:
        iterations: List of iteration numbers
        best_values: List of best values found so far
        save_path: Optional path to save the plot
        title: Plot title
        use_seed_rewards: Whether seed rewards were used for initialization
        n_seed_rewards: Number of seed rewards used (for plotting division)
    """
    plt.figure(figsize=(12, 6))
    
    if use_seed_rewards and n_seed_rewards > 0:
        # Split data into seed rewards and BO phases
        seed_iterations = iterations[:n_seed_rewards]
        seed_values = best_values[:n_seed_rewards]
        bo_iterations = iterations[n_seed_rewards-1:]  # Include last seed point for continuity
        bo_values = best_values[n_seed_rewards-1:]
        
        # Plot seed reward phase
        plt.plot(seed_iterations, seed_values, 'g-', linewidth=2, marker='s', 
                markersize=5, alpha=0.7, label=f'STARC Seed Rewards ({n_seed_rewards})')
        
        # Plot BO phase
        if len(bo_iterations) > 1:
            plt.plot(bo_iterations, bo_values, 'b-', linewidth=2, marker='o', 
                    markersize=4, alpha=0.8, label=f'Bayesian Optimization ({len(bo_iterations)-1})')
        
        # Add vertical line separating phases
        if len(iterations) > n_seed_rewards:
            plt.axvline(x=n_seed_rewards + 0.5, color='red', linestyle='--', alpha=0.6, 
                       label='BO Start')
            
        plt.legend()
        
    else:
        # Standard plot without seed rewards
        plt.plot(iterations, best_values, 'b-', linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Evaluation Number')
    plt.ylabel('Best Distance Traveled So Far')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add value annotations for key points
    if len(best_values) > 0:
        # Annotate first point
        plt.annotate(f'{best_values[0]:.2f}', 
                    (iterations[0], best_values[0]), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
        
        # Annotate final point
        plt.annotate(f'{best_values[-1]:.2f}', 
                    (iterations[-1], best_values[-1]), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
        
        # If using seed rewards, annotate best seed reward
        if use_seed_rewards and n_seed_rewards > 0 and n_seed_rewards < len(best_values):
            best_seed_value = best_values[n_seed_rewards-1]
            plt.annotate(f'Best Seed: {best_seed_value:.2f}', 
                        (n_seed_rewards, best_seed_value), 
                        textcoords="offset points", 
                        xytext=(10,0), ha='left', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Convergence plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_importance(feature_names: List[str],
                          weights: List[float],
                          save_path: Optional[str] = None,
                          title: str = "Feature Importance") -> None:
    """
    Plot feature importance based on weight magnitudes.
    
    Args:
        feature_names: List of feature names
        weights: List of feature weights
        save_path: Optional path to save the plot
        title: Plot title
    """
    # Filter active features (non-zero weights)
    active_features = [(name, weight) for name, weight in zip(feature_names, weights) 
                      if abs(weight) > 1e-6]
    
    if not active_features:
        print("[PLOT] No active features to plot")
        return
    
    # Sort by absolute weight magnitude
    active_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Limit to top 15 features for readability
    if len(active_features) > 15:
        active_features = active_features[:15]
    
    names, weights_vals = zip(*active_features)
    
    # Create colors based on sign
    colors = ['green' if w > 0 else 'red' for w in weights_vals]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(names)), weights_vals, color=colors, alpha=0.7)
    
    plt.yticks(range(len(names)), names)
    plt.xlabel('Weight')
    plt.ylabel('Features')
    plt.title(f"{title} (Top {len(active_features)} Features)")
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, weight in zip(bars, weights_vals):
        width = bar.get_width()
        plt.text(width + (0.02 if width > 0 else -0.02), bar.get_y() + bar.get_height()/2,
                f'{weight:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive'),
                      Patch(facecolor='red', alpha=0.7, label='Negative')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Feature importance plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(results_dict: dict, save_path: str) -> None:
    """
    Save optimization results to JSON file.
    
    Args:
        results_dict: Dictionary containing optimization results
        save_path: Path to save the JSON file
    """
    import json
    from pathlib import Path
    
    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"[SAVE] Results saved to: {save_path}") 