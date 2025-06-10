import IPython
import torch
import heapq
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import itertools

class TransposeType(Enum):
    MATRIX = "matrix"  # Swaps fastest changing dimensions
    BLOCK = "block"    # Swaps intermediate dimensions

@dataclass
class SwapOperation:
    """Represents a single swap operation"""
    dim1: int
    dim2: int
    groups: Tuple[Tuple[int, ...], ...]  # Grouped dimensions for the swap
    transpose_type: TransposeType
    cost: float
    
    def __str__(self):
        if len(self.groups) == 2 and len(self.groups[0]) == 1 and len(self.groups[1]) == 1:
            return f"Swap({self.dim1}, {self.dim2}) - {self.transpose_type.value} - cost: {self.cost:.2f}"
        else:
            return f"GroupSwap({self.groups}) - {self.transpose_type.value} - cost: {self.cost:.2f}"

class TensorState:
    """Represents the current state of tensor dimensions"""
    def __init__(self, dims: Tuple[int, ...], shape: Tuple[int, ...]):
        self.dims = dims
        self.shape = shape
        
    def __eq__(self, other):
        return self.dims == other.dims
    
    def __hash__(self):
        return hash(self.dims)
    
    def __lt__(self, other):
        """Enable comparison for heapq - compare by dims tuple"""
        return self.dims < other.dims
    
    def __str__(self):
        return f"TensorState(dims={self.dims}, shape={self.shape})"
    
    def copy(self):
        return TensorState(self.dims, self.shape)

class PermuteOptimizer:
    """Optimal decomposition of permute operations into transpose sequences"""
    
    def __init__(self, c1: float = 1.0, c2: float = 2.0):
        """
        Args:
            c1: Cost coefficient for matrix transpose operations
            c2: Cost coefficient for block transpose operations
        """
        self.c1 = c1  # Matrix transpose cost coefficient
        self.c2 = c2  # Block transpose cost coefficient
        
    def calculate_transpose_cost(self, state: TensorState, dim1: int, dim2: int, 
                               groups: Tuple[Tuple[int, ...], ...]) -> Tuple[float, TransposeType]:
        """Calculate the cost and type of a transpose operation"""
        # Calculate total bytes processed
        total_elements = 1
        for s in state.shape:
            total_elements *= s
        bytes_processed = total_elements * 4  # Assuming float32
        
        # Determine if this is a matrix or block transpose
        max_dim = len(state.dims) - 1
        
        # Check if we're swapping the last two dimensions (fastest changing)
        if (dim2 == max_dim and dim1 == max_dim - 1) or \
           (dim1 == max_dim and dim2 == max_dim - 1):
            return self.c1 * bytes_processed, TransposeType.MATRIX
        else:
            return self.c2 * bytes_processed, TransposeType.BLOCK
    
    def generate_adjacent_swaps(self, state: TensorState) -> List[SwapOperation]:
        """Generate all valid adjacent dimension swaps"""
        swaps = []
        n_dims = len(state.dims)
        
        # Generate single dimension swaps (adjacent only)
        for i in range(n_dims - 1):
            dim1, dim2 = i, i + 1
            groups = ((dim1,), (dim2,))
            cost, transpose_type = self.calculate_transpose_cost(state, dim1, dim2, groups)
            
            swap = SwapOperation(
                dim1=dim1,
                dim2=dim2,
                groups=groups,
                transpose_type=transpose_type,
                cost=cost
            )
            swaps.append(swap)
        
        # Generate grouped dimension swaps (adjacent groups only)
        for group_size1 in range(1, n_dims):
            for group_size2 in range(1, n_dims - group_size1 + 1):
                for start1 in range(n_dims - group_size1 - group_size2 + 1):
                    start2 = start1 + group_size1
                    
                    # Only consider adjacent groups where start2 immediately follows group1
                    if start2 == start1 + group_size1:
                        # Map the dimension indices to actual current dimensions
                        group1 = tuple(state.dims[start1 + i] for i in range(group_size1))
                        group2 = tuple(state.dims[start2 + i] for i in range(group_size2))
                        
                        # Skip single dimension swaps (already handled above)
                        if len(group1) == 1 and len(group2) == 1:
                            continue
                        
                        groups = (group1, group2)
                        # Use the first dimension of each group for cost calculation
                        cost, transpose_type = self.calculate_transpose_cost(
                            state, start1, start2, groups
                        )
                        
                        swap = SwapOperation(
                            dim1=start1,
                            dim2=start2,
                            groups=groups,
                            transpose_type=transpose_type,
                            cost=cost
                        )
                        swaps.append(swap)
        
        return swaps
    
    def apply_swap(self, state: TensorState, swap: SwapOperation) -> TensorState:
        """Apply a swap operation to create a new state"""
        new_dims = list(state.dims)
        new_shape = list(state.shape)
        
        # Extract the groups
        group1, group2 = swap.groups
        
        # Find positions of the groups in the current state
        group1_positions = []
        group2_positions = []
        
        for dim in group1:
            group1_positions.append(new_dims.index(dim))
        for dim in group2:
            group2_positions.append(new_dims.index(dim))
        
        # Sort positions
        group1_positions.sort()
        group2_positions.sort()
        
        # For adjacent groups, we can simply swap their contents
        # The positions should be contiguous
        all_positions = sorted(group1_positions + group2_positions)
        
        # Extract values at these positions
        group1_dims = [new_dims[pos] for pos in group1_positions]
        group1_shapes = [new_shape[pos] for pos in group1_positions]
        group2_dims = [new_dims[pos] for pos in group2_positions]
        group2_shapes = [new_shape[pos] for pos in group2_positions]
        
        # Debug: print what we're swapping
        # print(f"    Swapping groups: {group1} <-> {group2}")
        # print(f"    Group1 at positions {group1_positions}: {group1_dims}")
        # print(f"    Group2 at positions {group2_positions}: {group2_dims}")
        
        # Place group2 where group1 was, and group1 where group2 was
        # But we need to place them in the contiguous block
        new_values_in_block = group2_dims + group1_dims
        new_shapes_in_block = group2_shapes + group1_shapes
        
        # Update the positions
        for i, pos in enumerate(all_positions):
            new_dims[pos] = new_values_in_block[i]
            new_shape[pos] = new_shapes_in_block[i]
        
        # print(f"    Result: {tuple(new_dims)}")
        
        return TensorState(tuple(new_dims), tuple(new_shape))
    
    def dijkstra_search(self, initial_state: TensorState, target_state: TensorState, 
                       max_iterations: int = 1000) -> Tuple[List[SwapOperation], float]:
        """Find optimal sequence using Dijkstra's algorithm"""
        print(f"Starting Dijkstra search from {initial_state.dims} to {target_state.dims}")
        
        # Priority queue: (cost, unique_id, state, path)
        # Using unique_id to break ties and avoid state comparison
        pq = [(0.0, 0, initial_state, [])]
        visited: Set[Tuple[int, ...]] = set()
        best_cost: Dict[Tuple[int, ...], float] = {initial_state.dims: 0.0}
        
        iterations = 0
        unique_id = 1  # Counter for unique IDs
        
        while pq and iterations < max_iterations:
            current_cost, _, current_state, path = heapq.heappop(pq)
            iterations += 1
            
            if current_state.dims in visited:
                continue
                
            visited.add(current_state.dims)
            
            print(f"Iteration {iterations}: Exploring state {current_state.dims} with cost {current_cost:.2f}")
            
            # Check if we reached the target
            if current_state == target_state:
                print(f"Found optimal solution with cost {current_cost:.2f} in {iterations} iterations")
                print(f"Final state: dims={current_state.dims}, shape={current_state.shape}")
                print(f"Target state: dims={target_state.dims}, shape={target_state.shape}")
                return path, current_cost
            
            # Generate all possible swaps
            possible_swaps = self.generate_adjacent_swaps(current_state)
            
            for swap in possible_swaps:
                new_state = self.apply_swap(current_state, swap)
                new_cost = current_cost + swap.cost
                
                # Skip if we've seen this state with better cost
                if new_state.dims in best_cost and best_cost[new_state.dims] <= new_cost:
                    continue
                
                best_cost[new_state.dims] = new_cost
                new_path = path + [swap]
                
                print(f"  -> Considering swap: {swap}")
                print(f"     Results in state: {new_state.dims} with total cost: {new_cost:.2f}")
                
                heapq.heappush(pq, (new_cost, unique_id, new_state, new_path))
                unique_id += 1
        
        print(f"Search completed after {iterations} iterations")
        return [], float('inf')  # No solution found
    
    def optimize_permute(self, original_shape: Tuple[int, ...], 
                        target_permutation: Tuple[int, ...]) -> Tuple[List[SwapOperation], float]:
        """
        Find optimal sequence of swaps to achieve target permutation
        
        Args:
            original_shape: Shape of the original tensor
            target_permutation: Target permutation (e.g., (3, 0, 1, 2))
            
        Returns:
            Tuple of (optimal_swap_sequence, total_cost)
        """
        print(f"\n=== Optimizing permutation from {tuple(range(len(original_shape)))} to {target_permutation} ===")
        print(f"Tensor shape: {original_shape}")
        
        initial_state = TensorState(tuple(range(len(original_shape))), original_shape)
        target_state = TensorState(target_permutation, tuple(original_shape[i] for i in target_permutation))
        
        print(f"Initial state: dims={initial_state.dims}, shape={initial_state.shape}")
        print(f"Target state: dims={target_state.dims}, shape={target_state.shape}")
        
        return self.dijkstra_search(initial_state, target_state)

def execute_swap_sequence(tensor: torch.Tensor, swap_sequence: List[SwapOperation]) -> torch.Tensor:
    """Execute a sequence of swap operations on a tensor"""
    result = tensor.clone()
    current_dims = list(range(len(tensor.shape)))
    
    print(f"\nExecuting swap sequence on tensor with shape {tensor.shape}")
    print(f"Initial dimension order: {current_dims}")
    
    for i, swap in enumerate(swap_sequence):
        print(f"Step {i+1}: {swap}")
        print(f"  Before: shape={result.shape}, dims={current_dims}")
        
        # Apply the swap by constructing the new permutation
        group1, group2 = swap.groups
        
        # Find where these dimension values are currently located
        group1_current_positions = []
        group2_current_positions = []
        
        for dim_val in group1:
            group1_current_positions.append(current_dims.index(dim_val))
        for dim_val in group2:
            group2_current_positions.append(current_dims.index(dim_val))
        
        # Sort to maintain relative order within groups
        group1_current_positions.sort()
        group2_current_positions.sort()
        
        # Create new dimension ordering by swapping the groups
        new_dims = current_dims.copy()
        
        # Get the dimension values at the current positions
        group1_values = [current_dims[pos] for pos in group1_current_positions]
        group2_values = [current_dims[pos] for pos in group2_current_positions]
        
        print(f"    Group1 values: {group1_values} at positions {group1_current_positions}")
        print(f"    Group2 values: {group2_values} at positions {group2_current_positions}")
        
        # For adjacent group swap, we need to place group2 where group1 was and vice versa
        # But we need to be careful about the ordering
        
        # Find all affected positions (should be contiguous for adjacent swaps)
        all_positions = sorted(group1_current_positions + group2_current_positions)
        
        # The new arrangement should be group2_values followed by group1_values
        new_values_at_positions = group2_values + group1_values
        
        print(f"    All positions: {all_positions}")
        print(f"    New values: {new_values_at_positions}")
        
        # Update the positions
        for j, pos in enumerate(all_positions):
            new_dims[pos] = new_values_at_positions[j]
        
        print(f"    New dims arrangement: {new_dims}")
        
        # Calculate the permutation needed to transform current arrangement to new arrangement
        permutation = []
        for target_dim in new_dims:
            permutation.append(current_dims.index(target_dim))
        
        print(f"    Permutation to apply: {permutation}")
        
        result = result.permute(permutation)
        current_dims = new_dims
        
        print(f"  After:  shape={result.shape}, dims={current_dims}")
    
    return result

def test_permute_optimizer():
    """Test the permute optimizer with various cases"""
    print("=" * 80)
    print("TESTING PERMUTE OPTIMIZER")
    print("=" * 80)
    
    optimizer = PermuteOptimizer(c1=1.0, c2=2.0)
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple 2D transpose',
            'shape': (4, 3),
            'permutation': (1, 0)
        },
        {
            'name': 'Bubble sort example',
            'shape': (2, 3, 4, 5),
            'permutation': (3, 0, 1, 2)
        },
        {
            'name': '3D rotation',
            'shape': (2, 3, 4),
            'permutation': (2, 0, 1)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'='*60}")
        
        shape = test_case['shape']
        target_perm = test_case['permutation']
        
        # Manual verification of what the target should be
        original_tensor = torch.randn(shape)
        expected_result = original_tensor.permute(target_perm)
        print(f"Manual verification:")
        print(f"  Original shape: {original_tensor.shape}")
        print(f"  Target permutation: {target_perm}")
        print(f"  Expected result shape: {expected_result.shape}")
        
        # Find optimal sequence
        swap_sequence, total_cost = optimizer.optimize_permute(shape, target_perm)
        
        if swap_sequence:
            print(f"\nOptimal sequence found with total cost: {total_cost:.2f}")
            print("Swap sequence:")
            for i, swap in enumerate(swap_sequence, 1):
                print(f"  {i}. {swap}")
            
            # Test with PyTorch
            print(f"\nTesting with PyTorch:")
            original_tensor = torch.randn(shape)
            print(f"Original tensor shape: {original_tensor.shape}")
            
            # PyTorch reference
            pytorch_result = original_tensor.permute(target_perm)
            print(f"PyTorch permute result shape: {pytorch_result.shape}")
            print(f"PyTorch result dims: {target_perm}")
            
            # Our implementation
            our_result = execute_swap_sequence(original_tensor, swap_sequence)
            print(f"Our result shape: {our_result.shape}")
            
            # Let's also verify the final dimension ordering matches what we expect
            print(f"Expected final dims: {target_perm}")
            
            # Verify correctness
            if pytorch_result.shape == our_result.shape:
                if torch.allclose(pytorch_result, our_result, atol=1e-6):
                    print("✅ Results match PyTorch reference!")
                else:
                    print("❌ Shapes match but values differ!")
                    print(f"Max difference: {torch.max(torch.abs(pytorch_result - our_result))}")
            else:
                print("❌ Shapes do not match!")
                print(f"Expected shape: {pytorch_result.shape}")
                print(f"Got shape: {our_result.shape}")
        else:
            print("❌ No solution found!")
        
        # Test all cases to see how they perform
        # if test_case['name'] == '3D rotation':
        #     break

if __name__ == "__main__":
    test_permute_optimizer()
    IPython.embed()
