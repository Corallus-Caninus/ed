#In Memory of Oshkosh, my pet Dalmatian.
import torch
from typing import Optional, Union
from torch import Tensor
@torch.jit.script
class SparseFlatTensor:
    def __init__(self, starts, ends, values, total_size, unit_indices=None, unit_values=None):
        """
        Represents a 1D sparse tensor using start and end indices for sparse segments and unit indices.
        Args:
            starts (torch.Tensor): 1D tensor of start indices for each dense segment.
            ends (torch.Tensor): 1D tensor of end indices for each dense segment.
            values (torch.Tensor): 1D tensor containing concatenated values of all dense segments.
            total_size (Tensor): The total size of the 1D tensor.
            unit_indices (torch.Tensor, optional): 1D tensor of indices for unit elements. Defaults to None.
            unit_values (torch.Tensor, optional): 1D tensor of values for unit elements. Defaults to None.
        """
        self.starts = starts.to(torch.long)
        self.ends = ends.to(torch.long)
        self.values = values # Now a 1D tensor
        self.total_size = torch.tensor(total_size).to(torch.long)
        self.unit_indices = unit_indices.to(torch.long) if unit_indices is not None else torch.empty(0, dtype=torch.long, device=starts.device)
        self.unit_values = unit_values if unit_values is not None else torch.empty(0, dtype=values.dtype, device=starts.device)
    def pin_memory(self):
        """
        Pins the memory of all internal tensors if they are on CPU.
        Returns a new SparseFlatTensor with pinned tensors.
        """
        return SparseFlatTensor(
            self.starts.pin_memory(), self.ends.pin_memory(), self.values.pin_memory(),
            self.total_size.pin_memory(), self.unit_indices.pin_memory(), self.unit_values.pin_memory()
        )
    def __repr__(self):
        return f"SparseFlatTensor(starts={self.starts}, ends={self.ends}, values={self.values}, total_size={self.total_size}, unit_indices={self.unit_indices.numel()})"
    def to_dense(self):
        """
        Converts the sparse tensor representation to a dense PyTorch tensor.
        Uses vectorized operations for performance with millions of segments.
        """
        dense_tensor = torch.zeros(self.total_size, dtype=self.values.dtype, device=self.values.device)
        # Process segments using vectorized operations
        if self.starts.numel() > 0:
            # Calculate segment lengths
            segment_lengths = self.ends - self.starts
            # Create a tensor of all segment indices
            # First create offsets for each segment
            segment_offsets = torch.cat([
                torch.tensor([0], device=self.starts.device),
                segment_lengths.cumsum(0)[:-1]
            ])
            # Create a range tensor for all values
            value_indices = torch.arange(self.values.numel(), device=self.starts.device)
            # Calculate which segment each value belongs to
            segment_ids = torch.searchsorted(segment_lengths.cumsum(0), value_indices, right=True)
            # Calculate the global index for each value
            global_indices = self.starts[segment_ids] + (value_indices - segment_offsets[segment_ids])
            # Assign all segment values at once
            dense_tensor[global_indices] = self.values
        # Process unit indices
        if self.unit_indices.numel() > 0:
            dense_tensor[self.unit_indices] = self.unit_values
        return dense_tensor
    def to(self, device: torch.device, non_blocking: bool = False, pin_memory: bool = False):
        """
        Moves all internal tensors to the specified device and returns a new SparseFlatTensor, including unit indices.
        """
        return SparseFlatTensor(
            self.starts.to(device=device, dtype=self.starts.dtype, non_blocking=non_blocking),
            self.ends.to(device=device, dtype=self.ends.dtype, non_blocking=non_blocking),
            self.values.to(device=device, dtype=self.values.dtype, non_blocking=non_blocking),
            self.total_size.to(device=device, dtype=self.total_size.dtype, non_blocking=non_blocking),
            self.unit_indices.to(device=device, dtype=self.unit_indices.dtype, non_blocking=non_blocking),
            self.unit_values.to(device=device, dtype=self.unit_values.dtype, non_blocking=non_blocking)
        )
    def dot(self, other):
        """
        Computes the dot product of this SparseFlatTensor with another SparseFlatTensor, including unit indices.
        """
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        return torch.dot(dense_self, dense_other)
    def __mul__(self, scalar):
        """Scalar multiplication."""
        multiplied_values = self.values * scalar
        multiplied_unit_values = self.unit_values * scalar
        return SparseFlatTensor(
            self.starts, self.ends, multiplied_values, self.total_size,
            self.unit_indices, multiplied_unit_values
        )
    def __rmul__(self, other):
        """Right multiplication. Handles multiplication with dense tensors."""
        if isinstance(other, Tensor):
            # Element-wise multiplication with a dense tensor
            dense_self = self.to_dense()
            result = dense_self * other
            return result
        else:
            # Scalar multiplication
            return self.__mul__(other)
    def rmul(self, scalar):
        """Scalar multiplication."""
        return self.__mul__(scalar)
    def get_nonzero_mask(self):
        """
        Returns a boolean mask indicating non-zero positions in the sparse tensor.
        """
        mask = torch.zeros(self.total_size, dtype=torch.bool, device=self.values.device)
        # Process segments
        if self.starts.numel() > 0:
            segment_lengths = self.ends - self.starts
            segment_indices_offsets = torch.repeat_interleave(self.starts, segment_lengths)
            indices = torch.arange(segment_lengths.sum(), device=self.starts.device)
            segment_lengths_cumsum = segment_lengths.cumsum(0)
            start_indices = torch.cat([torch.tensor([0], device=self.starts.device), segment_lengths_cumsum[:-1]])
            segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
            segment_internal_indices = indices - start_indices[segment_ids]
            segment_indices = segment_indices_offsets + segment_internal_indices
            mask[segment_indices] = True
        # Process unit indices
        if self.unit_indices.numel() > 0:
            mask[self.unit_indices] = True
        return mask
    @staticmethod
    def add_sparse_dense(sparse_tensor: 'SparseFlatTensor', dense_tensor_arg: Tensor) -> Tensor:
        """Adds a SparseFlatTensor to a dense tensor in-place, including unit indices.
        Args:
            sparse_tensor (SparseFlatTensor): The sparse tensor to add.
            dense_tensor (Tensor): The dense tensor to add to.
        Returns:
            Tensor: The dense result of the addition.
        """
        dense_tensor = dense_tensor_arg # Explicitly use dense_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"
        result_dense_tensor = dense_tensor # Removed .clone() to make it in-place
        # Process segments
        if sparse_tensor.starts.numel() > 0:
            segment_lengths = sparse_tensor.ends - sparse_tensor.starts
            segment_indices_offsets = torch.repeat_interleave(sparse_tensor.starts, segment_lengths)
            indices = torch.arange(segment_lengths.sum(), device=sparse_tensor.starts.device)
            segment_lengths_cumsum = segment_lengths.cumsum(0)
            start_indices = torch.cat([torch.tensor([0], device=sparse_tensor.starts.device), segment_lengths_cumsum[:-1]])
            segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
            segment_internal_indices = indices - start_indices[segment_ids]
            segment_indices = segment_indices_offsets + segment_internal_indices
            result_dense_tensor.view(-1)[segment_indices] += sparse_tensor.values
        # Process unit indices
        if sparse_tensor.unit_indices.numel() > 0:
            result_dense_tensor.view(-1)[sparse_tensor.unit_indices] += sparse_tensor.unit_values
        return result_dense_tensor
    @staticmethod
    def _add_sparse_dense_alpha(sparse_tensor: 'SparseFlatTensor', dense_tensor_arg: Tensor, alpha: float = 1.0, offset: int = 0) -> Tensor:
        """
        Adds a SparseFlatTensor to a dense tensor, with an optional scaling factor alpha.
        The scaling is applied to the sparse tensor's values before addition.
        Indices are adjusted by the given offset.
        Args:
            sparse_tensor (SparseFlatTensor): The sparse tensor to add.
            dense_tensor_arg (Tensor): The dense tensor to add to.
            alpha (float, optional): Scaling factor for the sparse tensor's values. Defaults to 1.0.
            offset (int, optional): The starting index within the sparse tensor's global representation
                                   that corresponds to the start of dense_tensor_arg. Defaults to 0.
        Returns:
            Tensor: The dense result of the addition (dense_tensor_arg is modified in-place).
        """
        # dense_tensor = dense_tensor_arg # Explicitly use dense_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"
        # result_dense_tensor = dense_tensor.clone() # Not needed for in-place
        # Debug statements to identify bounds issue - removed for clarity, can be re-added if needed
        # print(f"dense_tensor size: {dense_tensor_arg.numel()}")
        # print(f"sparse_tensor total_size: {sparse_tensor.total_size}")
        # print(f"offset: {offset}")
        # if sparse_tensor.starts.numel() > 0:
        #     # ... (similar debug prints adjusted for offset logic)
        # print(f"max unit_indices (global): {sparse_tensor.unit_indices.max() if sparse_tensor.unit_indices.numel() > 0 else 'N/A'}")
        # Process segments
        if sparse_tensor.starts.numel() > 0:
            # --- Key Change: Adjust indices relative to the offset ---
            # Filter segments that potentially overlap with the current dense_tensor region
            # The dense tensor covers indices [offset, offset + dense_tensor_arg.numel())
            region_start = offset
            region_end = offset + dense_tensor_arg.numel()
            # Find segments that start before the region ends and end after the region starts
            potential_overlap_mask = (sparse_tensor.starts < region_end) & (sparse_tensor.ends > region_start)
            if potential_overlap_mask.any():
                filtered_starts = sparse_tensor.starts[potential_overlap_mask]
                filtered_ends = sparse_tensor.ends[potential_overlap_mask]
                original_segment_lengths = sparse_tensor.ends - sparse_tensor.starts
                original_value_starts = torch.cat([
                    torch.tensor([0], device=original_segment_lengths.device), original_segment_lengths.cumsum(0)[:-1]])
                # Vectorized segment processing
                seg_start_global = filtered_starts
                seg_end_global = filtered_ends
                seg_len = seg_end_global - seg_start_global
                # Generate all global indices for segments in a vectorized manner
                global_indices_for_segments = torch.repeat_interleave(seg_start_global, seg_len) + torch.arange(seg_len.sum(), device=sparse_tensor.values.device) - torch.repeat_interleave(torch.arange(seg_len.numel(), device=sparse_tensor.values.device) * seg_len, seg_len)
                # Adjust to local indices relative to the dense_tensor_arg's view
                local_indices_for_segments = global_indices_for_segments - offset
                # Check which of these local indices are valid (within dense_tensor_arg bounds)
                valid_mask_local = (local_indices_for_segments >= 0) & (local_indices_for_segments < dense_tensor_arg.numel())
                valid_local_indices = local_indices_for_segments[valid_mask_local]
                if valid_local_indices.numel() > 0:
                    # Get the original indices of the filtered segments
                    original_indices = torch.nonzero(potential_overlap_mask).squeeze(1)
                    # Get the corresponding values from the original values tensor in a vectorized manner
                    original_value_starts_filtered = original_value_starts[original_indices]
                    # Calculate indices for values, ensuring they stay within bounds
                    value_indices = original_value_starts_filtered.repeat_interleave(seg_len) + torch.arange(seg_len.sum(), device=sparse_tensor.values.device) - torch.repeat_interleave(torch.arange(seg_len.numel(), device=sparse_tensor.values.device) * seg_len, seg_len)
                    # Clip indices to stay within bounds of sparse_tensor.values
                    value_indices = torch.clamp(value_indices, 0, sparse_tensor.values.numel() - 1)
                    # Select only the values corresponding to valid indices
                    valid_values_for_segments = sparse_tensor.values[value_indices][valid_mask_local]
                    # Apply scaling
                    scaled_values_to_add = valid_values_for_segments * alpha
                    # Perform the in-place addition
                    dense_tensor_arg.view(-1)[valid_local_indices] += scaled_values_to_add
        # Process unit indices
        if sparse_tensor.unit_indices.numel() > 0:
            # --- Key Change: Adjust unit indices relative to the offset ---
            global_unit_indices = sparse_tensor.unit_indices
            local_unit_indices = global_unit_indices - offset
            # --- Key Change: Bounds check for local unit indices ---
            valid_unit_mask = (local_unit_indices >= 0) & (local_unit_indices < dense_tensor_arg.numel())
            if valid_unit_mask.any():
                final_local_unit_indices = local_unit_indices[valid_unit_mask]
                unit_values_to_add = sparse_tensor.unit_values[valid_unit_mask] * alpha # Apply scaling and filter
                dense_tensor_arg.view(-1)[final_local_unit_indices] += unit_values_to_add
        # Return the modified tensor (in-place modification)
        return dense_tensor_arg
    @staticmethod
    def sparse_dot_dense(sparse_tensor_arg: 'SparseFlatTensor', dense_tensor):
        """
        Computes the dot product of a SparseFlatTensor with a dense tensor, optimized for sparsity and unit indices.
        The sparse tensor's values are cast to the dtype of the dense tensor to avoid precision errors.
        """
        sparse_tensor = sparse_tensor_arg # Explicitly use sparse_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"
        # Cast sparse tensor values to match the dense tensor's dtype to avoid precision errors
        sparse_values = sparse_tensor.values.to(dense_tensor.dtype)
        unit_values = sparse_tensor.unit_values.to(dense_tensor.dtype)
        # Initialize dot product with unit indices contribution
        dot_product = torch.tensor(0.0, device=sparse_tensor.values.device, dtype=dense_tensor.dtype)
        if sparse_tensor.unit_indices.numel() > 0:
            unit_values_from_dense = dense_tensor.view(-1)[sparse_tensor.unit_indices]
            dot_product += torch.dot(unit_values_from_dense, unit_values)
        # Process segments if they exist
        if sparse_tensor.starts.numel() > 0:
            segment_lengths = sparse_tensor.ends - sparse_tensor.starts
            segment_indices_offsets = torch.repeat_interleave(sparse_tensor.starts, segment_lengths)
            indices = torch.arange(segment_lengths.sum(), device=sparse_tensor.starts.device)
            segment_lengths_cumsum = segment_lengths.cumsum(0)
            start_indices = torch.cat([torch.tensor([0], device=sparse_tensor.starts.device), segment_lengths_cumsum[:-1]])
            segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
            segment_internal_indices = indices - start_indices[segment_ids]
            segment_indices = segment_indices_offsets + segment_internal_indices
            # Extract corresponding values from dense tensor using sparse indices
            sparse_values_from_dense = dense_tensor.view(-1)[segment_indices]
            # Add segment contribution to dot product
            dot_product += torch.dot(sparse_values_from_dense, sparse_values)
        return dot_product

    @staticmethod
    def dense_to_sparse_flat_tensor(dense_tensor: Tensor):
        """
        Converts a dense tensor to SparseFlatTensor representation.
        """
        device = dense_tensor.device
        dtype = dense_tensor.dtype
        total_size = dense_tensor.numel()
        # Find indices of non-zero elements
        non_zero_indices = torch.nonzero(dense_tensor.view(-1)).squeeze()
        # Skip sparsification if tensor is already dense (all elements are non-zero)
        if non_zero_indices.numel() == total_size:
            # Return a "sparse" representation that's actually dense
            return SparseFlatTensor(
                torch.tensor([0], dtype=torch.int64, device=device),
                torch.tensor([total_size], dtype=torch.int64, device=device),
                dense_tensor.view(-1),
                total_size,
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=dtype, device=device)
            )
        if non_zero_indices.numel() == 0:  # Handle completely sparse tensor
            starts_local = torch.empty(0, dtype=torch.int64, device=device)
            ends_local = torch.empty(0, dtype=torch.int64, device=device)
            values_local = torch.empty(0, dtype=dtype, device=device) # Ensure dtype matches dense_tensor
            unit_indices_local = torch.empty(0, dtype=torch.long, device=device)
            unit_values_local = torch.empty(0, dtype=dtype, device=device)
            total_size_local = torch.tensor(total_size)
        else:
            # Find start and end indices of contiguous segments
            diff = non_zero_indices[1:] - non_zero_indices[:-1]
            segment_ends_indices = torch.nonzero(diff > 1).squeeze() + 1
            # Ensure segment_ends_indices is 1D for concatenation
            if segment_ends_indices.ndim == 0 and segment_ends_indices.numel() > 0:
                segment_ends_indices = segment_ends_indices.unsqueeze(0)
            elif segment_ends_indices.numel() == 0:
                segment_ends_indices = torch.empty(0, dtype=torch.long, device=device)
            # Correctly identify start and end indices in the non_zero_indices tensor
            # Indices in non_zero_indices for the start of each segment
            start_indices_in_non_zero = torch.cat([torch.tensor([0], dtype=torch.long, device=device), segment_ends_indices])
            # Indices in non_zero_indices for the end of each segment
            end_indices_in_non_zero = torch.cat([segment_ends_indices - 1, torch.tensor([len(non_zero_indices) - 1], dtype=torch.long, device=device)])
            # Actual start and end indices in the original flattened tensor
            starts_local_segments = non_zero_indices[start_indices_in_non_zero]
            # The end index should be the index *after* the last element of the segment.
            ends_local_segments = non_zero_indices[end_indices_in_non_zero] + 1
            segment_lengths = ends_local_segments - starts_local_segments
            # Identify unit-length segments
            is_unit_segment = (segment_lengths == 1)
            # Identify unit-length segments
            is_unit_segment = (segment_lengths == 1)
            # Indices in starts_local_segments/ends_local_segments that correspond to unit segments
            unit_segment_mask = is_unit_segment
            # Extract unit indices and values
            # The index of a unit segment is simply its start index
            unit_indices_local = starts_local_segments[unit_segment_mask]
            unit_values_local = dense_tensor.view(-1)[unit_indices_local]
            # Filter out unit segments to get actual segments
            segment_mask = ~is_unit_segment
            starts_local = starts_local_segments[segment_mask]
            ends_local = ends_local_segments[segment_mask]
            segment_lengths = segment_lengths[segment_mask] # Update segment_lengths for non-unit segments
            avg_segment_length = segment_lengths.float().mean() if segment_lengths.numel() > 0 else torch.tensor(0.0)
            max_segment_length = segment_lengths.max() if segment_lengths.numel() > 0 else torch.tensor(0)
            min_segment_length = segment_lengths.min() if segment_lengths.numel() > 0 else torch.tensor(0)
            print(f"Average segment length: {avg_segment_length:.4f}, Max segment length: {max_segment_length}, Min segment length: {min_segment_length}, Unit indices count: {unit_indices_local.numel()}, Segments count: {starts_local.numel()}")
            # Use direct indexing for better numerical stability
            # Get all non-zero indices for segments (excluding unit segments)
            if segment_mask.any():
                # Get segment ranges
                seg_starts = start_indices_in_non_zero[segment_mask]
                seg_ends = end_indices_in_non_zero[segment_mask]
                    
                # Vectorized segment index calculation without large intermediates
                segment_lengths_vals = seg_ends - seg_starts + 1
                total_segment_elements = segment_lengths_vals.sum()
                    
                # Global indices = segment_starts + intra-segment offsets
                segment_start_repeated = torch.repeat_interleave(seg_starts, segment_lengths_vals)
                intra_offsets = torch.arange(total_segment_elements, device=device) - torch.repeat_interleave(
                    torch.cat([torch.tensor([0], device=device), segment_lengths_vals.cumsum(0)[:-1]]), 
                    segment_lengths_vals
                )
                    
                segment_indices = non_zero_indices[segment_start_repeated + intra_offsets]
                values_local = dense_tensor.view(-1)[segment_indices]
                    
                # Clean up intermediates
                del segment_start_repeated, intra_offsets
            else:
                values_local = torch.empty(0, dtype=dtype, device=device)
            total_size_local = torch.tensor(total_size)
        # Create the sparse tensor
        sparse_result = SparseFlatTensor(starts_local, ends_local, values_local, total_size_local, unit_indices_local, unit_values_local)
    # TODO: debug this. There are discrepencies that dont seem to be from precision. Extract SparseFlatTensor and do some ATTD
    #    # Verify the sparse tensor matches the dense tensor
    #    dense_reconstruction = sparse_result.to_dense()
    #    diff = dense_tensor.view(-1) - dense_reconstruction
    #    max_diff = diff.abs().max().item()
    #    mean_diff = diff.abs().mean().item()
    #    non_zero_dense = (dense_tensor.view(-1) != 0).sum().item()
    #    non_zero_sparse = (dense_reconstruction != 0).sum().item()
    #    print(f"Sparse tensor verification:")
    #    print(f"  Max absolute difference: {max_diff}")
    #    print(f"  Mean absolute difference: {mean_diff}")
    #    print(f"  Non-zero in dense: {non_zero_dense}")
    #    print(f"  Non-zero in sparse reconstruction: {non_zero_sparse}")
    #    if max_diff > 1e-6:
    #        print(f"WARNING: Significant difference detected in sparse tensor conversion!")
        return sparse_result
