#!/usr/bin/env python3
"""
Bayesian Optimization for HalfCheetah reward function discovery.

This module implements BO to optimize canonical sum-of-features reward functions
for the HalfCheetah environment, with optional initialization from STARC seed rewards.
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import time

from .objective import HalfCheetahObjective
from .utils import plot_convergence, plot_feature_importance, save_results
from .load_existing_rewards import load_seed_rewards_as_initial_points, load_canonical_features, plot_seed_rewards


class BayesianOptimizer:
    """
    Bayesian optimizer for canonical reward function weights.
    
    Uses Gaussian Process surrogate modeling with acquisition function optimization
    to efficiently search the [-1, 1]^n weight space for optimal reward functions.
    Can initialize with STARC seed rewards or random Sobol points.
    """
    
    def __init__(self, 
                 objective: HalfCheetahObjective,
                 n_initial_points: int = 10,
                 acquisition_function: str = 'EI',
                 exploration_factor: float = 2.0,
                 device: Optional[str] = None,
                 results_dir: str = "bo_results",
                 use_seed_rewards: bool = False,
                 max_seed_rewards: Optional[int] = None,
                   bound_expansion: float = 2.0,
                   initial_total_points: Optional[int] = None,
                   discrete: bool = False,
                   discrete_decimals: int = 2):
        """
        Initialize Bayesian optimizer.
        
        Args:
            objective: Reward optimization objective function
            n_initial_points: Number of initial points (ignored if use_seed_rewards=True)
            acquisition_function: 'UCB', 'EI', or 'PI'
            exploration_factor: Exploration parameter (beta for UCB, higher = more exploration)
            device: PyTorch device ('cpu', 'cuda', etc.)
            results_dir: Directory to save optimization results
            use_seed_rewards: Whether to use STARC seed rewards as initial points
            max_seed_rewards: Maximum number of seed rewards to use (None for all)
            bound_expansion: How much to expand bounds beyond STARC ranges (default: 2.0)
            initial_total_points: If using seed rewards, augment with Sobol points to reach this count
        """
        self.objective = objective
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.exploration_factor = exploration_factor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.use_seed_rewards = use_seed_rewards
        self.max_seed_rewards = max_seed_rewards
        self.bound_expansion = bound_expansion
        self.initial_total_points = initial_total_points
        self.discrete = discrete
        self.discrete_decimals = max(0, int(discrete_decimals))
        
        # Get problem dimensions
        self.n_features = objective.get_feature_count()
        print(f"[BO] Optimizing {self.n_features} feature weights")
        
        # Initialize bounds (will be set properly in generate_initial_points)
        if use_seed_rewards:
            # Dynamic bounds based on original STARC weight ranges ±bound_expansion
            self.bounds = None  # Will be set in _load_seed_reward_points
            bounds_desc = f"Dynamic bounds per feature (original STARC ranges ±{bound_expansion})"
        else:
            # Fixed bounds for Sobol initialization  
            bounds_array = np.array([
                [-1.0] * self.n_features,  # Lower bounds
                [1.0] * self.n_features    # Upper bounds  
            ])
            self.bounds = torch.from_numpy(bounds_array).to(dtype=torch.float64, device=self.device)
            bounds_desc = "Fixed bounds [-1, 1] for all features"
        
        # Initialize storage
        self.X_observed = torch.empty(0, self.n_features, dtype=torch.float64, device=self.device)
        self.Y_observed = torch.empty(0, 1, dtype=torch.float64, device=self.device)
        self.evaluation_history = []
        
        # Store information about seed rewards if used
        self.seed_reward_names = []
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[BO] {bounds_desc}")
        print(f"[BO] Using {acquisition_function} acquisition with exploration_factor={exploration_factor}")
        if self.discrete:
            step = 10 ** (-self.discrete_decimals)
            print(f"[BO] Discrete search enabled: rounding to {self.discrete_decimals} decimals (step={step})")
        print(f"[BO] Device: {self.device}")
        print(f"[BO] Initial points: {'STARC seed rewards' if use_seed_rewards else f'{n_initial_points} Sobol points'}")

    def generate_initial_points(self, n_points: int) -> torch.Tensor:
        """
        Generate initial points using either STARC seed rewards or Sobol sequence.
        
        Args:
            n_points: Number of initial points to generate (ignored if using seed rewards)
            
        Returns:
            Tensor of initial points in [-1, 1]^n_features
        """
        if self.use_seed_rewards:
            return self._load_seed_reward_points()
        else:
            return self._generate_sobol_points(n_points)
    
    def _load_seed_reward_points(self) -> torch.Tensor:
        """Load STARC seed reward functions as initial points."""
        
        try:
            # Load seed rewards with dynamic bounds
            initial_points, reward_names, dynamic_bounds = load_seed_rewards_as_initial_points(
                max_rewards=self.max_seed_rewards,
                bound_expansion=self.bound_expansion
            )
            
            # Use dynamic bounds based on original STARC weight ranges
            self.bounds = dynamic_bounds.to(device=self.device)
            bounds_cpu = self.bounds.cpu().numpy()
            print(f"[BO] Using dynamic bounds per feature:")
            print(f"[BO] Min bounds: [{bounds_cpu[0].min():.2f}, {bounds_cpu[0].max():.2f}]")
            print(f"[BO] Max bounds: [{bounds_cpu[1].min():.2f}, {bounds_cpu[1].max():.2f}]")
            
            # Store reward names for analysis
            self.seed_reward_names = reward_names
            
            # Create plot of seed rewards
            canonical_features = load_canonical_features()
            plot_save_path = self.results_dir / "seed_rewards_initial_population.png"
            plot_seed_rewards(
                initial_points, 
                reward_names, 
                canonical_features,
                save_path=str(plot_save_path)
            )
            
            # Optionally augment with Sobol points to reach desired initial_total_points
            if self.initial_total_points is not None and len(initial_points) < self.initial_total_points:
                needed = self.initial_total_points - len(initial_points)
                print(f"[BO] Augmenting initial set with {needed} Sobol points (to reach {self.initial_total_points})")
                extra = self._generate_sobol_points(needed).cpu().numpy()
                initial_points = list(initial_points) + [row for row in extra]
                self.seed_reward_names = list(self.seed_reward_names) + [f"sobol_{i+1}" for i in range(needed)]

            # Convert to torch tensor
            initial_tensor = torch.tensor(
                np.array(initial_points), 
                dtype=torch.float64, 
                device=self.device
            )
            # Quantize to grid if discrete mode
            if self.discrete:
                initial_tensor = self._quantize_points(initial_tensor)
            
            print(f"[BO] ✓ Using {len(initial_points)} seed rewards with dynamic bounds per feature")
            print(f"[BO] ✓ Plot saved to: {plot_save_path}")
            
            # Show bounds summary
            bounds_cpu = self.bounds.cpu().numpy()
            print(f"[BO] ✓ Bounds per feature: [{bounds_cpu[0].min():.3f}, {bounds_cpu[0].max():.3f}] to [{bounds_cpu[1].min():.3f}, {bounds_cpu[1].max():.3f}]")
            
            return initial_tensor
            
        except Exception as e:
            print(f"[BO] ⚠️  Failed to load seed rewards: {e}")
            print(f"[BO] Falling back to Sobol initialization with fixed bounds [-1, 1]...")
            # Set fixed bounds as fallback
            bounds_array = np.array([
                [-1.0] * self.n_features,
                [1.0] * self.n_features
            ])
            self.bounds = torch.from_numpy(bounds_array).to(dtype=torch.float64, device=self.device)
            return self._generate_sobol_points(self.n_initial_points)
    
    def _generate_sobol_points(self, n_points: int) -> torch.Tensor:
        """Generate initial points using Sobol sequence."""
        print(f"[BO] Generating {n_points} initial points using Sobol sequence")
        
        # Generate Sobol points in [0, 1]^n_features
        sobol = SobolEngine(dimension=self.n_features, scramble=True)
        sobol_points = sobol.draw(n_points)
        
        # Transform to [-1, 1]^n_features
        initial_points = 2.0 * sobol_points - 1.0
        initial_points = initial_points.to(device=self.device, dtype=torch.float64)
        
        # If discrete mode, quantize to grid respecting bounds
        if self.discrete:
            initial_points = self._quantize_points(initial_points)
        
        return initial_points
    
    def evaluate_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Evaluate a batch of candidate points.
        
        Args:
            candidates: Tensor of candidate points (n_candidates x n_features)
            
        Returns:
            Tensor of objective values (n_candidates x 1)
        """
        values = []
        
        for i, candidate in enumerate(candidates):
            print(f"[BO] Evaluating candidate {i+1}/{len(candidates)}")
            
            # Convert to numpy and evaluate
            # If discrete mode, ensure candidate is quantized before evaluation
            if self.discrete:
                candidate = self._quantize_points(candidate.view(1, -1)).view(-1)
            weights_np = candidate.detach().cpu().numpy()
            start_time = time.time()
            
            try:
                value = self.objective.evaluate(weights_np)
                eval_time = time.time() - start_time
                
                # Determine if this is a seed reward
                is_seed_reward = (i < len(self.seed_reward_names) and 
                                len(self.evaluation_history) < len(self.seed_reward_names))
                seed_name = self.seed_reward_names[i] if is_seed_reward else None
                
                # Store evaluation info
                eval_info = {
                    'weights': weights_np.tolist(),
                    'value': float(value),
                    'evaluation_time': eval_time,
                    'timestamp': datetime.now().isoformat(),
                    'active_features': int(np.sum(np.abs(weights_np) > 1e-6)),
                    'weight_magnitude': float(np.linalg.norm(weights_np)),
                    'is_seed_reward': is_seed_reward,
                    'seed_name': seed_name
                }
                self.evaluation_history.append(eval_info)
                
                # Print result with seed info
                if is_seed_reward:
                    print(f"[BO] Result: {value:.4f} (SEED: {seed_name}, active: {eval_info['active_features']}, "
                          f"magnitude: {eval_info['weight_magnitude']:.3f}, time: {eval_time:.1f}s)")
                else:
                    print(f"[BO] Result: {value:.4f} (active: {eval_info['active_features']}, "
                          f"magnitude: {eval_info['weight_magnitude']:.3f}, time: {eval_time:.1f}s)")
                
                values.append(value)
                
            except Exception as e:
                print(f"[BO] Evaluation failed: {e}")
                values.append(float('-inf'))  # Handle failed evaluations
        
        return torch.tensor(values, dtype=torch.float64, device=self.device).unsqueeze(-1)
    
    def fit_gaussian_process(self) -> SingleTaskGP:
        """
        Fit Gaussian Process model to observed data.
        
        Returns:
            Fitted GP model
        """
        print(f"[BO] Fitting GP model with {len(self.X_observed)} observations")
        
        try:
            # Create and fit GP model with standard transforms for stability
            gp_model = SingleTaskGP(
                train_X=self.X_observed,
                train_Y=self.Y_observed,
                input_transform=Normalize(d=self.n_features),
                outcome_transform=Standardize(m=1),
            )
            
            # Fit the model
            mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
            fit_gpytorch_model(mll)
            
            return gp_model
            
        except Exception as e:
            print(f"[BO] Warning: GP fitting failed: {e}")
            print(f"[BO] Attempting fallback GP fitting...")
            
            try:
                # Try with standardized Y values
                Y_std = (self.Y_observed - self.Y_observed.mean()) / (self.Y_observed.std() + 1e-6)
                gp_model = SingleTaskGP(
                    train_X=self.X_observed,
                    train_Y=Y_std
                )
                mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
                fit_gpytorch_model(mll)
                return gp_model
                
            except Exception as e2:
                print(f"[BO] Error: Both GP fitting attempts failed: {e2}")
                raise RuntimeError(f"Could not fit GP model: {e}, {e2}")

    def get_acquisition_function(self, gp_model: SingleTaskGP):
        """Get acquisition function for optimization."""
        try:
            if self.acquisition_function == 'UCB':
                # Analytic UCB does not require a sampler
                return UpperConfidenceBound(
                    model=gp_model,
                    beta=self.exploration_factor
                )
            elif self.acquisition_function == 'EI':
                best_value = self.Y_observed.max()
                from botorch.acquisition.analytic import ExpectedImprovement
                return ExpectedImprovement(
                    model=gp_model,
                    best_f=best_value
                )
            else:
                raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
                
        except Exception as e:
            print(f"[BO] Warning: Acquisition function creation failed: {e}")
            # Fallback to UCB with lower beta
            print(f"[BO] Falling back to UCB with reduced exploration")
            return UpperConfidenceBound(
                model=gp_model,
                beta=1.0  # Reduced exploration factor
            )
    
    def optimize_acquisition(self, acquisition_function) -> torch.Tensor:
        """
        Optimize acquisition function to get next candidate.
        
        Args:
            acquisition_function: Acquisition function to optimize
            
        Returns:
            Next candidate point
        """
        try:
            candidate, _ = optimize_acqf(
                acq_function=acquisition_function,
                bounds=self.bounds,
                q=1,  # Single point acquisition
                num_restarts=40,
                raw_samples=1024,
            )
            # Quantize to discrete grid if enabled, and ensure uniqueness
            if self.discrete:
                candidate = self._quantize_points(candidate)
                # If candidate duplicates an observed point due to rounding, jitter minimally on-grid
                if self._is_duplicate(candidate):
                    candidate = self._next_on_grid(candidate)
            
            return candidate
            
        except Exception as e:
            print(f"[BO] Warning: Acquisition optimization failed: {e}")
            print(f"[BO] Falling back to random sampling within bounds")
            
            # Fallback: random sampling within bounds
            bounds_cpu = self.bounds.cpu().numpy()
            random_point = np.random.uniform(
                bounds_cpu[0], 
                bounds_cpu[1], 
                size=(1, self.n_features)
            )
            tensor_point = torch.tensor(random_point, dtype=torch.float64, device=self.device)
            if self.discrete:
                tensor_point = self._quantize_points(tensor_point)
            return tensor_point
    
    def run_optimization(self, n_iterations: int = 50) -> Dict:
        """
        Run the full Bayesian optimization loop.
        
        Args:
            n_iterations: Number of BO iterations (after initial points)
            
        Returns:
            Dictionary with optimization results
        """
        initial_count = len(self.seed_reward_names) if self.use_seed_rewards else self.n_initial_points
        
        print(f"[BO] Starting optimization with {n_iterations} iterations")
        print(f"[BO] Total evaluations will be: {initial_count + n_iterations}")
        
        start_time = time.time()
        
        # Phase 1: Evaluate initial points
        print("\n" + "="*60)
        if self.use_seed_rewards:
            print("PHASE 1: EVALUATING STARC SEED REWARDS")
        else:
            print("PHASE 1: INITIAL EXPLORATION")
        print("="*60)
        
        initial_points = self.generate_initial_points(self.n_initial_points)
        initial_values = self.evaluate_candidates(initial_points)
        
        # Store initial observations
        self.X_observed = initial_points
        self.Y_observed = initial_values
        
        best_idx = torch.argmax(self.Y_observed)
        best_value = self.Y_observed[best_idx].item()
        best_weights = self.X_observed[best_idx].cpu().numpy()
        
        # Show seed reward results if applicable
        if self.use_seed_rewards:
            print(f"\n[BO] Seed reward evaluation complete!")
            print(f"[BO] Best seed reward value: {best_value:.4f}")
            
            # Show top 3 seed rewards
            seed_values = list(zip(self.seed_reward_names, initial_values.cpu().numpy().flatten()))
            seed_values.sort(key=lambda x: x[1], reverse=True)
            print(f"[BO] Top 3 seed rewards:")
            for i, (name, value) in enumerate(seed_values[:3]):
                print(f"[BO]   {i+1}. {name}: {value:.4f}")
        else:
            print(f"\n[BO] Initial exploration complete!")
            print(f"[BO] Best initial value: {best_value:.4f}")
        
        print(f"[BO] Best weights: {best_weights}")
        
        # SAVE INITIAL RESULTS (in case experiment crashes later)
        self._save_intermediate_results(iteration=0, n_iterations=n_iterations)
        
        # Phase 2: Bayesian optimization loop
        for iteration in range(n_iterations):
            try:
                print(f"\n--- BO Iteration {iteration + 1}/{n_iterations} ---")
                
                # Fit GP model to all observed data
                gp_model = self.fit_gaussian_process()
                
                # Get acquisition function
                acquisition_function = self.get_acquisition_function(gp_model)
                
                # Optimize acquisition function to get next candidate
                next_candidate = self.optimize_acquisition(acquisition_function)
                
                # Evaluate the new candidate
                next_value = self.evaluate_candidates(next_candidate)
                
                # Update observations
                self.X_observed = torch.cat([self.X_observed, next_candidate])
                self.Y_observed = torch.cat([self.Y_observed, next_value])
                
                # Update best if improved
                current_best_idx = torch.argmax(self.Y_observed)
                current_best_value = self.Y_observed[current_best_idx].item()
                current_best_weights = self.X_observed[current_best_idx].cpu().numpy()
                
                if current_best_value > best_value:
                    improvement = current_best_value - best_value
                    best_value = current_best_value
                    best_weights = current_best_weights
                    print(f"[BO] 🎉 NEW BEST! Value: {best_value:.4f} (improvement: +{improvement:.4f})")
                else:
                    print(f"[BO] Current best remains: {best_value:.4f}")
                
                # Show progress
                total_evaluations = len(self.X_observed)
                elapsed_time = time.time() - start_time
                print(f"[BO] Progress: {total_evaluations}/{initial_count + n_iterations} "
                      f"({elapsed_time:.1f}s elapsed)")
                
                # INCREMENTAL SAVE every 10 iterations and on final iteration
                if (iteration + 1) % 10 == 0 or iteration == n_iterations - 1:
                    print(f"[BO] 💾 Saving intermediate results (iteration {iteration + 1})")
                    self._save_intermediate_results(iteration=iteration + 1, n_iterations=n_iterations)
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  [BO] Keyboard interrupt received at iteration {iteration + 1}")
                print(f"[BO] Saving current results before stopping...")
                
                # Save current state
                self._save_intermediate_results(iteration=iteration, n_iterations=n_iterations)
                
                # Update final results with current state
                best_idx = torch.argmax(self.Y_observed)
                best_value = self.Y_observed[best_idx].item()
                best_weights = self.X_observed[best_idx].cpu().numpy()
                
                print(f"[BO] Experiment stopped at iteration {iteration + 1}/{n_iterations}")
                print(f"[BO] Current best value: {best_value:.4f}")
                print(f"[BO] Results saved. You can analyze partial results or resume later.")
                
                # Set flag to exit the loop
                n_iterations = iteration  # This will make the summary correct
                break
                
            except Exception as e:
                print(f"[BO] ⚠️  Error in iteration {iteration + 1}: {e}")
                print(f"[BO] Saving current results and attempting to continue...")
                
                # Save current state
                self._save_intermediate_results(iteration=iteration, n_iterations=n_iterations)
                
                # Continue to next iteration
                continue
        
        # Final results
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        print(f"[BO] Total time: {total_time:.1f}s")
        print(f"[BO] Total evaluations: {len(self.X_observed)}")
        print(f"[BO] Best value found: {best_value:.4f}")
        print(f"[BO] Active features in best: {np.sum(np.abs(best_weights) > 1e-6)}")
        print(f"[BO] Best weights magnitude: {np.linalg.norm(best_weights):.3f}")
        
        # Prepare results
        results = {
            'best_value': best_value,
            'best_weights': best_weights.tolist(),
            'best_features': self.objective.get_active_features(best_weights),
            'optimization_time': total_time,
            'total_evaluations': len(self.X_observed),
            'n_initial_points': initial_count,
            'n_bo_iterations': n_iterations,
            'acquisition_function': self.acquisition_function,
            'exploration_factor': self.exploration_factor,
            'bounds': self.bounds.cpu().numpy().tolist() if self.bounds is not None else None,
            'bounds_type': 'dynamic_seed_based' if self.use_seed_rewards else 'fixed',
            'use_seed_rewards': self.use_seed_rewards,
            'seed_reward_names': self.seed_reward_names,
            'evaluation_history': self.evaluation_history,
            'convergence_data': {
                'iterations': list(range(1, len(self.Y_observed) + 1)),
                'values': self.Y_observed.cpu().numpy().flatten().tolist(),
                'best_so_far': [max(self.Y_observed[:i+1].cpu().numpy().flatten()) 
                               for i in range(len(self.Y_observed))]
            }
        }
        
        # Save results
        self.save_results(results)
        
        return results

    # -------------------- Discretization helpers --------------------
    def _quantize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Round points to a fixed decimal grid and clip to bounds."""
        if points.ndim == 1:
            points = points.view(1, -1)
        scale = 10 ** self.discrete_decimals
        lb = self.bounds[0].to(device=points.device, dtype=points.dtype)
        ub = self.bounds[1].to(device=points.device, dtype=points.dtype)
        # Compute on-grid min/max within bounds per-dimension
        grid_min = torch.ceil(lb * scale) / scale
        grid_max = torch.floor(ub * scale) / scale
        # Round to grid
        rounded = torch.round(points * scale) / scale
        # Clamp to on-grid bounds
        clamped = torch.minimum(torch.maximum(rounded, grid_min), grid_max)
        return clamped

    def _is_duplicate(self, candidate: torch.Tensor) -> bool:
        """Check if quantized candidate duplicates any observed X (exact match)."""
        if candidate.ndim == 2:
            candidate = candidate.view(-1)
        if self.X_observed.numel() == 0:
            return False
        # Compare to all observed points after quantization
        cand = candidate.view(1, -1)
        diff = torch.abs(self.X_observed - cand)
        return bool(torch.any(torch.all(diff <= 0.0, dim=1)).item())

    def _next_on_grid(self, candidate: torch.Tensor) -> torch.Tensor:
        """Move candidate to the nearest next grid point within bounds to avoid duplicates."""
        if candidate.ndim == 2:
            candidate = candidate.view(-1)
        scale = 10 ** self.discrete_decimals
        lb = self.bounds[0]
        ub = self.bounds[1]
        # Try incremental steps along dimensions until a unique point is found
        step = 1.0 / scale
        for dim in range(self.n_features):
            up = candidate.clone()
            up[dim] = torch.minimum(up[dim] + step, ub[dim])
            up = self._quantize_points(up)
            if not self._is_duplicate(up):
                return up
            down = candidate.clone()
            down[dim] = torch.maximum(down[dim] - step, lb[dim])
            down = self._quantize_points(down)
            if not self._is_duplicate(down):
                return down
        # As a last resort, random on-grid within bounds
        rand = torch.empty_like(candidate)
        for dim in range(self.n_features):
            grid_min = torch.ceil(lb[dim] * scale) / scale
            grid_max = torch.floor(ub[dim] * scale) / scale
            if grid_max < grid_min:
                grid_max = grid_min
            # Sample integer grid between [min,max] inclusive
            n_steps = int(round((grid_max - grid_min) * scale)) + 1
            idx = torch.randint(low=0, high=max(1, n_steps), size=(1,)).item()
            rand[dim] = grid_min + idx * (1.0 / scale)
        rand = self._quantize_points(rand)
        return rand.view(1, -1)
    
    def save_results(self, results: Dict):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = self.results_dir / f"bo_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[BO] Results saved to {results_file}")
        
        # Generate plots
        try:
            # Convergence plot
            plot_convergence(
                results['convergence_data']['iterations'],
                results['convergence_data']['best_so_far'],
                save_path=self.results_dir / f"convergence_{timestamp}.png",
                use_seed_rewards=self.use_seed_rewards,
                n_seed_rewards=len(self.seed_reward_names) if self.use_seed_rewards else 0
            )
            
            # Feature importance plot
            if len(results['best_weights']) > 0:
                feature_names = list(self.objective.get_feature_names())
                plot_feature_importance(
                    feature_names,
                    results['best_weights'],
                    save_path=self.results_dir / f"feature_importance_{timestamp}.png"
                )
            
        except Exception as e:
            print(f"[BO] Warning: Could not generate plots: {e}")
    
    def _save_intermediate_results(self, iteration: int, n_iterations: int):
        """Save intermediate results during optimization to prevent data loss."""
        try:
            # Get current best
            if len(self.Y_observed) == 0:
                return
                
            best_idx = torch.argmax(self.Y_observed)
            best_value = self.Y_observed[best_idx].item()
            best_weights = self.X_observed[best_idx].cpu().numpy()
            
            # Create intermediate results
            intermediate_results = {
                'current_best_value': best_value,
                'current_best_weights': best_weights.tolist(),
                'total_evaluations': len(self.X_observed),
                'completed_iterations': iteration,
                'target_iterations': n_iterations,
                'use_seed_rewards': self.use_seed_rewards,
                'seed_reward_names': self.seed_reward_names,
                'evaluation_history': self.evaluation_history,
                'convergence_data': {
                    'iterations': list(range(1, len(self.Y_observed) + 1)),
                    'values': self.Y_observed.cpu().numpy().flatten().tolist(),
                    'best_so_far': [max(self.Y_observed[:i+1].cpu().numpy().flatten()) 
                                   for i in range(len(self.Y_observed))]
                },
                'bounds': self.bounds.cpu().numpy().tolist() if self.bounds is not None else None,
                'bounds_type': 'dynamic_seed_based' if self.use_seed_rewards else 'fixed',
                'timestamp': datetime.now().isoformat(),
                'status': 'in_progress' if iteration < n_iterations else 'completed'
            }
            
            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = self.results_dir / f"bo_intermediate_iter{iteration:03d}_{timestamp}.json"
            
            with open(intermediate_file, 'w') as f:
                json.dump(intermediate_results, f, indent=2)
            
            print(f"[BO] 💾 Intermediate results saved to {intermediate_file}")
            
            # Also save as "latest" for easy recovery
            latest_file = self.results_dir / "bo_latest_checkpoint.json"
            with open(latest_file, 'w') as f:
                json.dump(intermediate_results, f, indent=2)
                
        except Exception as e:
            print(f"[BO] Warning: Could not save intermediate results: {e}")
    
    def get_best_result(self) -> Tuple[float, np.ndarray]:
        """
        Get the best result found so far.
        
        Returns:
            Tuple of (best_value, best_weights)
        """
        if len(self.Y_observed) == 0:
            raise RuntimeError("No evaluations have been performed yet")
        
        best_idx = torch.argmax(self.Y_observed)
        best_value = self.Y_observed[best_idx].item()
        best_weights = self.X_observed[best_idx].cpu().numpy()
        
        return best_value, best_weights
    
    def get_optimization_summary(self) -> Dict:
        """Get a summary of the optimization progress."""
        if len(self.Y_observed) == 0:
            return {"status": "not_started"}
        
        best_value, best_weights = self.get_best_result()
        expected_initial = len(self.seed_reward_names) if self.use_seed_rewards else self.n_initial_points
        
        return {
            "status": "running" if len(self.Y_observed) < expected_initial else "completed",
            "evaluations_done": len(self.Y_observed),
            "best_value": best_value,
            "best_weights": best_weights.tolist(),
            "active_features": int(np.sum(np.abs(best_weights) > 1e-6)),
            "use_seed_rewards": self.use_seed_rewards,
            "seed_reward_count": len(self.seed_reward_names),
            "improvement_over_baseline": best_value - (self.Y_observed[0].item() if len(self.Y_observed) > 0 else 0)
        } 


 