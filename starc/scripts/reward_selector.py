import numpy as np
from typing import List, Dict, Tuple
from starc.config import STARCConfig

class RewardSelector:
    def __init__(self):
        pass
    
    def select_best_worst(self, starc_results: Dict, n_closest: int = 4, n_furthest: int = 4) -> Dict:
        """Select the closest and furthest rewards from ground truth"""
        
        reward_names = starc_results['reward_names']
        gt_distances = starc_results['ground_truth_distances']
        
        # Create list of (name, distance) pairs
        name_distance_pairs = list(zip(reward_names, gt_distances))
        
        # Sort by distance to ground truth
        sorted_pairs = sorted(name_distance_pairs, key=lambda x: x[1])
        
        # Select closest and furthest
        closest = [name for name, _ in sorted_pairs[:n_closest]]
        furthest = [name for name, _ in sorted_pairs[-n_furthest:]]
        
        # Get the actual distances for reporting
        closest_distances = [dist for _, dist in sorted_pairs[:n_closest]]
        furthest_distances = [dist for _, dist in sorted_pairs[-n_furthest:]]
        
        return {
            'closest': closest,
            'furthest': furthest,
            'closest_distances': closest_distances,
            'furthest_distances': furthest_distances,
            'all_sorted': sorted_pairs
        }
    
    def select_cluster_aware_best_worst(self, starc_results: Dict, clusters: Dict, 
                                      n_closest: int = None, n_furthest: int = None) -> Dict:
        """
        Cluster-aware selection that ensures diversity while respecting performance preferences.
        
        Strategy:
        1. For exploitation (closest): Prefer best performers but ensure cluster diversity
        2. For exploration (furthest): Prefer diverse representatives from different clusters
        3. Fallback to performance-based selection if not enough diversity
        """
        
        # Use config defaults if not specified
        if n_closest is None:
            n_closest = STARCConfig.N_CLOSEST_REWARDS
        if n_furthest is None:
            n_furthest = STARCConfig.N_FURTHEST_REWARDS
        
        reward_names = starc_results['reward_names']
        gt_distances = starc_results['ground_truth_distances']
        distance_matrix = starc_results['distance_matrix']
        
        # Create name to index and distance mappings
        name_to_idx = {name: idx for idx, name in enumerate(reward_names)}
        name_to_gt_dist = {name: dist for name, dist in zip(reward_names, gt_distances)}
        
        # Step 1: Get cluster representatives (1-2 per cluster based on size)
        cluster_representatives = self._select_balanced_cluster_representatives(
            clusters, name_to_gt_dist, distance_matrix, name_to_idx
        )
        
        # Step 2: Select exploitation set (closest to ground truth)
        exploitation_candidates = []
        for cluster_id, reps in cluster_representatives.items():
            # Sort cluster reps by ground truth distance (best first)
            sorted_reps = sorted(reps, key=lambda name: name_to_gt_dist[name])
            exploitation_candidates.extend(sorted_reps)
        
        # Take top performers with diversity preference
        closest = self._select_diverse_subset(
            exploitation_candidates, name_to_gt_dist, distance_matrix, 
            name_to_idx, n_closest, prefer_performance=True, minimize_distance=True
        )
        
        # Step 3: Select exploration set (furthest from ground truth)
        exploration_candidates = []
        for cluster_id, reps in cluster_representatives.items():
            # Sort cluster reps by ground truth distance (worst first for exploration)
            sorted_reps = sorted(reps, key=lambda name: name_to_gt_dist[name], reverse=True)
            exploration_candidates.extend(sorted_reps)
        
        # Remove already selected closest to avoid overlap
        exploration_candidates = [name for name in exploration_candidates if name not in closest]
        
        furthest = self._select_diverse_subset(
            exploration_candidates, name_to_gt_dist, distance_matrix,
            name_to_idx, n_furthest, prefer_performance=True, minimize_distance=False
        )
        
        # Step 4: Fallback if we don't have enough candidates
        if len(closest) < n_closest or len(furthest) < n_furthest:
            print(f"    ⚠️  Cluster-aware selection incomplete ({len(closest)}/{n_closest} closest, {len(furthest)}/{n_furthest} furthest)")
            print(f"    🔄 Falling back to performance-based selection...")
            return self.select_best_worst(starc_results, n_closest, n_furthest)
        
        # Get distances for reporting
        closest_distances = [name_to_gt_dist[name] for name in closest]
        furthest_distances = [name_to_gt_dist[name] for name in furthest]
        
        return {
            'closest': closest,
            'furthest': furthest,
            'closest_distances': closest_distances,
            'furthest_distances': furthest_distances,
            'cluster_representatives': cluster_representatives,
            'selection_method': 'cluster_aware'
        }
    
    def _select_balanced_cluster_representatives(self, clusters: Dict, name_to_gt_dist: Dict,
                                               distance_matrix: np.ndarray, name_to_idx: Dict) -> Dict:
        """
        Select 1-2 representatives per cluster based on cluster size and diversity.
        
        Strategy:
        - Small clusters (1-2 members): Take all
        - Medium clusters (3-4 members): Take 2 most diverse
        - Large clusters (5+ members): Take 2 most diverse, prefer different performance levels
        """
        
        representatives = {}
        
        for cluster_id, cluster_members in clusters.items():
            cluster_size = len(cluster_members)
            
            if cluster_size <= 2:
                # Small cluster: take all members
                representatives[cluster_id] = cluster_members
                
            elif cluster_size <= 4:
                # Medium cluster: take 2 most diverse
                representatives[cluster_id] = self._select_diverse_from_cluster(
                    cluster_members, distance_matrix, name_to_idx, n_select=2
                )
                
            else:
                # Large cluster: take 2 diverse with different performance levels
                representatives[cluster_id] = self._select_diverse_performance_from_cluster(
                    cluster_members, distance_matrix, name_to_idx, name_to_gt_dist, n_select=2
                )
        
        return representatives
    
    def _select_diverse_from_cluster(self, cluster_members: List[str], distance_matrix: np.ndarray,
                                   name_to_idx: Dict, n_select: int) -> List[str]:
        """Select most diverse members from a cluster"""
        
        if len(cluster_members) <= n_select:
            return cluster_members
        
        # Get cluster indices and distance submatrix
        cluster_indices = [name_to_idx[name] for name in cluster_members]
        cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        
        # Use greedy diverse selection
        selected_indices = self._select_diverse_greedy(cluster_distances, n_select)
        return [cluster_members[i] for i in selected_indices]
    
    def _select_diverse_performance_from_cluster(self, cluster_members: List[str], 
                                               distance_matrix: np.ndarray, name_to_idx: Dict,
                                               name_to_gt_dist: Dict, n_select: int) -> List[str]:
        """
        Select diverse members with different performance levels.
        Tries to get both a good performer and a poor performer for diversity.
        """
        
        if len(cluster_members) <= n_select:
            return cluster_members
        
        # Sort by ground truth performance
        sorted_members = sorted(cluster_members, key=lambda name: name_to_gt_dist[name])
        
        if n_select == 2:
            # Take best and worst, but check if they're diverse enough
            best = sorted_members[0]
            worst = sorted_members[-1]
            
            best_idx = name_to_idx[best]
            worst_idx = name_to_idx[worst]
            
            if distance_matrix[best_idx, worst_idx] >= STARCConfig.CLUSTER_DIVERSITY_THRESHOLD:
                return [best, worst]
            else:
                # Fall back to pure diversity selection
                return self._select_diverse_from_cluster(cluster_members, distance_matrix, name_to_idx, n_select)
        else:
            # For n_select > 2, use pure diversity
            return self._select_diverse_from_cluster(cluster_members, distance_matrix, name_to_idx, n_select)
    
    def _select_diverse_subset(self, candidates: List[str], name_to_gt_dist: Dict,
                             distance_matrix: np.ndarray, name_to_idx: Dict, n_select: int,
                             prefer_performance: bool = True, minimize_distance: bool = True) -> List[str]:
        """
        Select a diverse subset that balances performance and diversity.
        
        Args:
            candidates: List of candidate reward names
            prefer_performance: If True, bias towards better performers
            minimize_distance: If True, prefer closer to ground truth; if False, prefer further
        """
        
        if len(candidates) <= n_select:
            return candidates
        
        if prefer_performance:
            # Sort by performance (best first if minimize_distance, worst first otherwise)
            sorted_candidates = sorted(candidates, key=lambda name: name_to_gt_dist[name], 
                                     reverse=not minimize_distance)
            
            # Take top performers but ensure diversity
            selected = []
            
            for candidate in sorted_candidates:
                if len(selected) >= n_select:
                    break
                
                # Check diversity with already selected
                candidate_idx = name_to_idx[candidate]
                is_diverse = True
                
                for selected_name in selected:
                    selected_idx = name_to_idx[selected_name]
                    if distance_matrix[candidate_idx, selected_idx] < STARCConfig.SELECTION_DIVERSITY_THRESHOLD:
                        is_diverse = False
                        break
                
                if is_diverse or len(selected) == 0:
                    selected.append(candidate)
            
            # If we don't have enough, fill with remaining best performers
            if len(selected) < n_select:
                remaining = [name for name in sorted_candidates if name not in selected]
                selected.extend(remaining[:n_select - len(selected)])
            
            return selected[:n_select]
        
        else:
            # Pure diversity selection
            candidate_indices = [name_to_idx[name] for name in candidates]
            candidate_distances = distance_matrix[np.ix_(candidate_indices, candidate_indices)]
            selected_indices = self._select_diverse_greedy(candidate_distances, n_select)
            return [candidates[i] for i in selected_indices]
    
    def select_diverse_representatives(self, starc_results: Dict, clusters: Dict, n_per_cluster: int = 1) -> Dict:
        """Select diverse representatives from each cluster"""
        
        reward_names = starc_results['reward_names']
        distance_matrix = starc_results['distance_matrix']
        
        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(reward_names)}
        
        representatives = {}
        
        for cluster_id, cluster_members in clusters.items():
            if len(cluster_members) <= n_per_cluster:
                # If cluster is small, take all members
                representatives[cluster_id] = cluster_members
            else:
                # Select most diverse members within cluster
                cluster_indices = [name_to_idx[name] for name in cluster_members]
                
                # Extract submatrix for this cluster
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Select diverse representatives using greedy algorithm
                selected_indices = self._select_diverse_greedy(cluster_distances, n_per_cluster)
                selected_names = [cluster_members[i] for i in selected_indices]
                
                representatives[cluster_id] = selected_names
        
        return representatives
    
    def _select_diverse_greedy(self, distance_matrix: np.ndarray, n_select: int) -> List[int]:
        """Greedy algorithm to select diverse points"""
        n_points = distance_matrix.shape[0]
        
        if n_select >= n_points:
            return list(range(n_points))
        
        selected = []
        
        # Start with the point that has maximum average distance to all others
        avg_distances = np.mean(distance_matrix, axis=1)
        first_idx = np.argmax(avg_distances)
        selected.append(first_idx)
        
        # Greedily add points that are furthest from already selected points
        for _ in range(n_select - 1):
            remaining = [i for i in range(n_points) if i not in selected]
            
            best_idx = None
            best_min_distance = -1
            
            for candidate in remaining:
                # Find minimum distance to already selected points
                min_distance = min(distance_matrix[candidate, selected_idx] for selected_idx in selected)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = candidate
            
            if best_idx is not None:
                selected.append(best_idx)
        
        return selected
    
    def analyze_selection_quality(self, selected_rewards: Dict, starc_results: Dict) -> Dict:
        """Analyze the quality of the selected rewards"""
        
        closest = selected_rewards['closest']
        furthest = selected_rewards['furthest']
        
        reward_names = starc_results['reward_names']
        distance_matrix = starc_results['distance_matrix']
        gt_distances = starc_results['ground_truth_distances']
        
        # Create name to index mapping
        name_to_idx = {name: idx for idx, name in enumerate(reward_names)}
        
        analysis = {
            'closest_analysis': self._analyze_group(closest, name_to_idx, distance_matrix, gt_distances),
            'furthest_analysis': self._analyze_group(furthest, name_to_idx, distance_matrix, gt_distances),
            'diversity_between_groups': self._analyze_between_groups(closest, furthest, name_to_idx, distance_matrix)
        }
        
        return analysis
    
    def _analyze_group(self, group: List[str], name_to_idx: Dict, distance_matrix: np.ndarray, gt_distances: List[float]) -> Dict:
        """Analyze diversity and characteristics of a group"""
        
        if len(group) < 2:
            return {
                'size': len(group),
                'avg_internal_distance': 0.0,
                'avg_gt_distance': gt_distances[name_to_idx[group[0]]] if group else 0.0,
                'diversity_score': 0.0
            }
        
        # Get indices for this group
        indices = [name_to_idx[name] for name in group]
        
        # Calculate internal distances
        internal_distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                internal_distances.append(distance_matrix[indices[i], indices[j]])
        
        # Calculate ground truth distances
        group_gt_distances = [gt_distances[name_to_idx[name]] for name in group]
        
        return {
            'size': len(group),
            'avg_internal_distance': np.mean(internal_distances),
            'max_internal_distance': np.max(internal_distances),
            'min_internal_distance': np.min(internal_distances),
            'avg_gt_distance': np.mean(group_gt_distances),
            'std_gt_distance': np.std(group_gt_distances),
            'diversity_score': np.mean(internal_distances)  # Higher = more diverse
        }
    
    def _analyze_between_groups(self, group1: List[str], group2: List[str], name_to_idx: Dict, distance_matrix: np.ndarray) -> Dict:
        """Analyze distances between two groups"""
        
        indices1 = [name_to_idx[name] for name in group1]
        indices2 = [name_to_idx[name] for name in group2]
        
        between_distances = []
        for i in indices1:
            for j in indices2:
                between_distances.append(distance_matrix[i, j])
        
        return {
            'avg_between_distance': np.mean(between_distances),
            'max_between_distance': np.max(between_distances),
            'min_between_distance': np.min(between_distances),
            'std_between_distance': np.std(between_distances)
        } 