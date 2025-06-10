import pdb
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import heapq

@dataclass
class DimensionGroup:
    """Represents a group of consecutive dimensions."""
    dims: List[int]  # The actual dimension indices in this group
    
    def __repr__(self):
        if len(self.dims) == 1:
            return str(self.dims[0])
        return f"({', '.join(str(d) for d in self.dims)})"
    
    def __lt__(self, other):
        """Comparison for sorting."""
        return self.dims < other.dims
    
    def __eq__(self, other):
        """Equality comparison."""
        return self.dims == other.dims
    
    def __hash__(self):
        """Hash for use in sets."""
        return hash(tuple(self.dims))

@dataclass
class PermuteState:
    """Represents the current state of dimension arrangement."""
    permutation: List[int]  # Current permutation of dimensions
    
    def __repr__(self):
        return f"{self.permutation}"
    
    def to_tuple(self):
        """Convert to tuple for hashing."""
        return tuple(self.permutation)
    
    def __lt__(self, other):
        """Comparison for heap operations."""
        return self.permutation < other.permutation
    
    def __eq__(self, other):
        """Equality comparison."""
        return self.permutation == other.permutation
    
    def __hash__(self):
        """Hash for use in sets."""
        return hash(self.to_tuple())

@dataclass
class SwapOperation:
    """Represents a swap operation between adjacent dimension groups."""
    left_pos: int  # Starting position of left group
    left_size: int  # Size of left group
    right_size: int  # Size of right group
    cost: float
    operation_type: str  # 'matrix_transpose' or 'block_transpose'
    left_dims: List[int] = None  # Actual dimensions in left group
    right_dims: List[int] = None  # Actual dimensions in right group
    
    def __repr__(self):
        if self.left_dims and self.right_dims:
            left_str = str(self.left_dims[0]) if len(self.left_dims) == 1 else f"({', '.join(map(str, self.left_dims))})"
            right_str = str(self.right_dims[0]) if len(self.right_dims) == 1 else f"({', '.join(map(str, self.right_dims))})"
            return f"Swap {left_str} <-> {right_str} (type: {self.operation_type}, cost: {self.cost})"
        return f"Swap positions [{self.left_pos}:{self.left_pos+self.left_size}] <-> [{self.left_pos+self.left_size}:{self.left_pos+self.left_size+self.right_size}] (type: {self.operation_type}, cost: {self.cost})"
    
    def __lt__(self, other):
        """Comparison for sorting."""
        return (self.cost, self.left_pos, self.left_size, self.right_size) < \
               (other.cost, other.left_pos, other.left_size, other.right_size)
    
    def __eq__(self, other):
        """Equality comparison."""
        return (self.left_pos == other.left_pos and 
                self.left_size == other.left_size and
                self.right_size == other.right_size and
                self.cost == other.cost and
                self.operation_type == other.operation_type)
    
    def __hash__(self):
        """Hash for use in sets."""
        return hash((self.left_pos, self.left_size, self.right_size, self.cost, self.operation_type))

class PermuteDecomposer:
    def __init__(self, c1: float = 1.0, c2: float = 2.0):
        self.c1 = c1  # Cost factor for matrix_transpose
        self.c2 = c2  # Cost factor for block_transpose
        
    def matrix_transpose_cost(self, num_bytes: int) -> float:
        """Cost of matrix transpose operation."""
        return self.c1 * num_bytes
    
    def block_transpose_cost(self, num_bytes: int) -> float:
        """Cost of block transpose operation."""
        return self.c2 * num_bytes
    
    def matrix_transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        """Swap the two fastest-changing dimensions of a 3D tensor."""
        assert tensor.ndim == 3, "matrix_transpose requires 3D tensor"
        return tensor.transpose(-2, -1)
    
    def block_transpose(self, tensor: torch.Tensor) -> torch.Tensor:
        """Swap the middle two dimensions of a 4D tensor."""
        assert tensor.ndim == 4, "block_transpose requires 4D tensor"
        return tensor.transpose(1, 2)
    
    def execute_swap(self, tensor: torch.Tensor, swap: SwapOperation) -> torch.Tensor:
        """Execute a single swap operation on the tensor."""
        ndim = tensor.ndim
        left_start = swap.left_pos
        left_end = swap.left_pos + swap.left_size
        right_end = swap.left_pos + swap.left_size + swap.right_size
        
        # Calculate dimensions before and after the swap region
        dims_before = left_start
        dims_after = ndim - right_end
        
        if swap.operation_type == 'matrix_transpose':
            # Reshape to 3D: [before, left, right]
            shape_before = tensor.shape[:dims_before] if dims_before > 0 else ()
            shape_left = tensor.shape[left_start:left_end]
            shape_right = tensor.shape[left_end:right_end]
            shape_after = tensor.shape[right_end:] if dims_after > 0 else ()
            
            # Calculate sizes for 3D reshape
            size_before = np.prod(shape_before) if dims_before > 0 else 1
            size_left = np.prod(shape_left)
            size_right = np.prod(shape_right)
            
            # Reshape to 3D
            tensor_3d = tensor.reshape(size_before, size_left, size_right)
            
            # Transpose
            transposed_3d = self.matrix_transpose(tensor_3d)
            
            # Reshape back
            new_shape = shape_before + shape_right + shape_left + shape_after
            return transposed_3d.reshape(new_shape)
            
        else:  # block_transpose
            # Reshape to 4D: [before, left, right, after]
            shape_before = tensor.shape[:dims_before] if dims_before > 0 else ()
            shape_left = tensor.shape[left_start:left_end]
            shape_right = tensor.shape[left_end:right_end]
            shape_after = tensor.shape[right_end:] if dims_after > 0 else ()
            
            # Calculate sizes for 4D reshape
            size_before = np.prod(shape_before) if dims_before > 0 else 1
            size_left = np.prod(shape_left)
            size_right = np.prod(shape_right)
            size_after = np.prod(shape_after) if dims_after > 0 else 1
            
            # Reshape to 4D
            tensor_4d = tensor.reshape(size_before, size_left, size_right, size_after)
            
            # Transpose
            transposed_4d = self.block_transpose(tensor_4d)
            
            # Reshape back
            new_shape = shape_before + shape_right + shape_left + shape_after
            return transposed_4d.reshape(new_shape)
    
    def get_possible_swaps(self, state: PermuteState, tensor_shape: List[int]) -> List[Tuple[SwapOperation, PermuteState]]:
        """Get all possible swap operations from current state."""
        swaps = []
        perm = state.permutation
        ndim = len(perm)
        
        # Try all possible adjacent group swaps
        for left_start in range(ndim - 1):
            for left_size in range(1, ndim - left_start):
                for right_size in range(1, ndim - left_start - left_size + 1):
                    if left_start + left_size + right_size > ndim:
                        continue
                    
                    # Get the actual dimensions being swapped
                    left_dims = perm[left_start:left_start + left_size]
                    right_dims = perm[left_start + left_size:left_start + left_size + right_size]
                    
                    # Calculate dimensions before and after
                    dims_before = left_start
                    dims_after = ndim - (left_start + left_size + right_size)
                    
                    # Determine operation type and cost
                    num_bytes = np.prod(tensor_shape) * 4  # Assuming float32
                    
                    if dims_after == 0:
                        # Can use matrix transpose
                        cost = self.matrix_transpose_cost(num_bytes)
                        op_type = 'matrix_transpose'
                    else:
                        # Must use block transpose
                        cost = self.block_transpose_cost(num_bytes)
                        op_type = 'block_transpose'
                    
                    # Create swap operation
                    swap = SwapOperation(left_start, left_size, right_size, cost, op_type)
                    # Store the actual dimensions being swapped
                    swap.left_dims = left_dims
                    swap.right_dims = right_dims
                    
                    # Create new permutation after swap
                    new_perm = perm.copy()
                    new_perm[left_start:left_start + left_size + right_size] = right_dims + left_dims
                    
                    new_state = PermuteState(new_perm)
                    swaps.append((swap, new_state))
        
        return swaps
    
    def find_optimal_sequence(self, source_perm: List[int], target_perm: List[int], 
                            tensor_shape: List[int]) -> Tuple[List[SwapOperation], float]:
        """Find optimal sequence of swaps using Dijkstra's algorithm with pruning."""
        # Initialize states
        source_state = PermuteState(source_perm)
        target_tuple = tuple(target_perm)
        
        # Priority queue: (cost, state, path)
        pq = [(0, source_state, [])]
        visited = set()
        
        # Track best known cost to target (for pruning)
        best_cost_to_target = float('inf')
        
        print(f"Starting search from {source_state} to {target_perm}")
        print(f"Tensor shape: {tensor_shape}")
        print("-" * 80)
        
        iteration = 0
        while pq:
            current_cost, current_state, path = heapq.heappop(pq)
            
            # Pruning: skip if this path cannot beat the best known solution
            if current_cost >= best_cost_to_target:
                continue
            
            state_tuple = current_state.to_tuple()
            if state_tuple in visited:
                continue
            visited.add(state_tuple)
            
            # Check if we reached the target
            if state_tuple == target_tuple:
                best_cost_to_target = current_cost
                print(f"\nFound optimal path with cost {current_cost}:")
                for i, swap in enumerate(path):
                    print(f"  Step {i+1}: {swap}")
                return path, current_cost
            
            # Explore neighbors
            possible_swaps = self.get_possible_swaps(current_state, tensor_shape)
            
            print(f"\nExploring state: {current_state} (cost so far: {current_cost})")
            print(f"  Possible swaps: {len(possible_swaps)}")
            
            for swap, next_state in possible_swaps:
                next_cost = current_cost + swap.cost
                
                print(f"{iteration*'-'}-> {next_state} (cost: {swap.cost}, total: {next_cost})")
                
                # Pruning: skip if this path is already too expensive
                if next_cost >= best_cost_to_target:
                    print('**Pruned**')
                    continue
                
                new_path = path + [swap]
                
                heapq.heappush(pq, (next_cost, next_state, new_path))
            
            iteration += 1
        
        raise ValueError("No path found!")
    
    def decompose_permute(self, tensor: torch.Tensor, target_perm: List[int]) -> torch.Tensor:
        """Decompose and execute a permute operation optimally."""
        source_perm = list(range(tensor.ndim))
        tensor_shape = list(tensor.shape)
        
        # Find optimal sequence
        swap_sequence, total_cost = self.find_optimal_sequence(source_perm, target_perm, tensor_shape)
        
        print(f"\nTotal cost: {total_cost}")
        print(f"Number of operations: {len(swap_sequence)}")
        
        # Execute the sequence
        result = tensor
        current_perm = source_perm.copy()
        
        for i, swap in enumerate(swap_sequence):
            print(f"\nExecuting step {i+1}: {swap}")
            
            # Execute the swap on the tensor
            result = self.execute_swap(result, swap)
            
            # Update current permutation to track progress
            left_dims = current_perm[swap.left_pos:swap.left_pos + swap.left_size]
            right_dims = current_perm[swap.left_pos + swap.left_size:swap.left_pos + swap.left_size + swap.right_size]
            current_perm[swap.left_pos:swap.left_pos + swap.left_size + swap.right_size] = right_dims + left_dims
            
            print(f"  New permutation: {current_perm}")
        
        return result


def test_decomposer():
    """Test the permute decomposer with various examples."""
    decomposer = PermuteDecomposer(c1=1.0, c2=2.0)
    
    # Test case 1: Simple permutation
    print("=" * 80)
    print("Test Case 1: Simple permutation (0, 1, 2, 3) -> (3, 0, 1, 2)")
    print("=" * 80)
    
    tensor1 = torch.randn(2, 3, 4, 5)
    target_perm1 = [3, 0, 1, 2]
    
    result1 = decomposer.decompose_permute(tensor1, target_perm1)
    expected1 = tensor1.permute(target_perm1)
    pdb.set_trace()
    
    print(f"\nVerification: torch.allclose(result, expected) = {torch.allclose(result1, expected1)}")
    print(f"Max difference: {(result1 - expected1).abs().max().item()}")
    
    # Test case 2: More complex permutation
    print("\n" + "=" * 80)
    print("Test Case 2: Complex permutation (0, 1, 2, 3, 4) -> (2, 4, 0, 3, 1)")
    print("=" * 80)
    
    tensor2 = torch.randn(2, 3, 4, 5, 6)
    target_perm2 = [2, 4, 0, 3, 1]
    
    result2 = decomposer.decompose_permute(tensor2, target_perm2)
    expected2 = tensor2.permute(target_perm2)
    
    print(f"\nVerification: torch.allclose(result, expected) = {torch.allclose(result2, expected2)}")
    print(f"Max difference: {(result2 - expected2).abs().max().item()}")
    
    # Test case 3: Reverse permutation
    print("\n" + "=" * 80)
    print("Test Case 3: Reverse permutation (0, 1, 2) -> (2, 1, 0)")
    print("=" * 80)
    
    tensor3 = torch.randn(4, 5, 6)
    target_perm3 = [2, 1, 0]
    
    result3 = decomposer.decompose_permute(tensor3, target_perm3)
    expected3 = tensor3.permute(target_perm3)
    
    print(f"\nVerification: torch.allclose(result, expected) = {torch.allclose(result3, expected3)}")
    print(f"Max difference: {(result3 - expected3).abs().max().item()}")


if __name__ == "__main__":
    test_decomposer()
