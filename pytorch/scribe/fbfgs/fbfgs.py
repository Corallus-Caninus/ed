#In Memory of Oshkosh, my pet Dalmatian.
import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
from typing import Optional, Union
from torch import device
import torch
from torch import Tensor
import gc
import psutil
import time
import torch.distributed as dist
import sys
from concurrent.futures import ThreadPoolExecutor

#def trace_calls(frame, event, arg):
#    if event == 'line':
#        if 'gtd' in frame.f_locals or 'gtd_new' in frame.f_locals:
#            print(f"gtd: {frame.f_locals.get('gtd', 'N/A')}, gtd_new: {frame.f_locals.get('gtd_new', 'N/A')}")
#    return trace_calls
#
#sys.settrace(trace_calls)

from torch.optim.optimizer import Optimizer, ParamsT

#TODO: ensure we are memory efficient. Gather Grads should replace the grads with the view. Im not sure about the implementation but at least we wont allocate a lot of indices for the views? this should not take as much memory as CUDA is saying it does so theres a lot of stuff that can be GC optimized
#TODO: distribution: need to also distributed the norm. Write our own l1 and turn norm hyperparam into a scalar coefficient to ensure the l1 is stable for networks with high parameter count and low type precision.
#TODO: implement SparseFlatTensor addition correctly via AI rendering
#TODO: extract this to a module and begin FBFGS project structuring
#TODO: if we have 1 segment (we havent induced sparsity) its probably worth it to do a dense computation both in terms of memory and compute resources.
#TODO: WE STILL HAVE PRECISION ERRORS IN THESE OPS. gtd calculated from the dense was not the same as gtd calculated from the sparse. This could be preventing gaps and causing a lot of other errors.
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
            segment_ids = torch.searchsorted(segment_lengths.cumsum(0), value_indices, right=False)

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
                original_value_starts = torch.cat([torch.tensor([0], device=original_segment_lengths.device), original_segment_lengths.cumsum(0)[:-1]])

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
    def _add_sparse_dense(sparse_tensor: 'SparseFlatTensor', dense_tensor_arg: Tensor, offset: int = 0) -> Tensor:
        """
        Adds a SparseFlatTensor to a dense tensor in-place.
        Indices are adjusted by the given offset.

        Args:
            sparse_tensor (SparseFlatTensor): The sparse tensor to add.
            dense_tensor_arg (Tensor): The dense tensor to add to. This tensor will be modified.
            offset (int, optional): The starting index within the sparse tensor's global representation
                                   that corresponds to the start of dense_tensor_arg. Defaults to 0.

        Returns:
            Tensor: The modified dense tensor (dense_tensor_arg).
        """
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"

        # Process segments
        if sparse_tensor.starts.numel() > 0:
            # --- Key Change: Adjust indices relative to the offset ---
            # Filter segments that potentially overlap with the current dense_tensor region
            region_start = offset
            region_end = offset + dense_tensor_arg.numel()
            potential_overlap_mask = (sparse_tensor.starts < region_end) & (sparse_tensor.ends > region_start)

            if potential_overlap_mask.any():
                filtered_starts = sparse_tensor.starts[potential_overlap_mask]
                filtered_ends = sparse_tensor.ends[potential_overlap_mask]
                original_segment_lengths = sparse_tensor.ends - sparse_tensor.starts
                original_value_starts = torch.cat([torch.tensor([0], device=original_segment_lengths.device), original_segment_lengths.cumsum(0)[:-1]])

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

                    # Perform the in-place addition
                    dense_tensor_arg.view(-1)[valid_local_indices] += valid_values_for_segments

        # Process unit indices
        if sparse_tensor.unit_indices.numel() > 0:
            # --- Key Change: Adjust unit indices relative to the offset ---
            global_unit_indices = sparse_tensor.unit_indices
            local_unit_indices = global_unit_indices - offset

            # --- Key Change: Bounds check for local unit indices ---
            valid_unit_mask = (local_unit_indices >= 0) & (local_unit_indices < dense_tensor_arg.numel())
            if valid_unit_mask.any():
                final_local_unit_indices = local_unit_indices[valid_unit_mask]
                # No alpha scaling here, direct addition
                dense_tensor_arg.view(-1)[final_local_unit_indices] += sparse_tensor.unit_values[valid_unit_mask]

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

__all__ = ["FBFGS"]

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

    # Verify the sparse tensor matches the dense tensor
    dense_reconstruction = sparse_result.to_dense()
    diff = dense_tensor.view(-1) - dense_reconstruction
    max_diff = diff.abs().max().item()
    mean_diff = diff.abs().mean().item()
    non_zero_dense = (dense_tensor.view(-1) != 0).sum().item()
    non_zero_sparse = (dense_reconstruction != 0).sum().item()

#    print(f"Sparse tensor verification:")
#    print(f"  Max absolute difference: {max_diff}")
#    print(f"  Mean absolute difference: {mean_diff}")
#    print(f"  Non-zero in dense: {non_zero_dense}")
#    print(f"  Non-zero in sparse reconstruction: {non_zero_sparse}")

#    if max_diff > 1e-6:
#        print(f"WARNING: Significant difference detected in sparse tensor conversion!")

    return sparse_result

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        min_pos_tensor = torch.tensor(min_pos)
        xmin_bound_tensor = torch.tensor(xmin_bound, device=min_pos_tensor.device)
        xmax_bound_tensor = torch.tensor(xmax_bound, device=min_pos_tensor.device)
        return min(max(min_pos_tensor, xmin_bound_tensor), xmax_bound_tensor)
    else:
#TODO: this is bad we can do much better. in zoom phase this shouldnt matter but this can retard bracket phase.
        return torch.tensor((xmin_bound + xmax_bound) / 2.0, device=g1.device)

#TODO: on relaxed wolfe, if loss is reduced from the previous iteration of this data point, accept it (the first iteration is the relaxed wolfe).
#TODO: c3 along with armijo that is c2 but for overconvergence? To prevent early convergence on insta-wolfes? Probably not necessary and would probably slow things down #TODO: cleanup all the AI device mess
def _strong_wolfe(
    obj_func, direction_device, t, d, f, g, gtd, c1=1e-20, c2=0.9, tolerance_change=1e-16, max_ls=5, bracket_shift=(1/3), bracket_shove=(1/3), capture_min_step=1e-4, capture_max_step=100, optimizer_device: str = 'cuda'):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
#    g_best g.to(direction_device)
    f_new, g_new = obj_func(t, d)
#TODO: better solution for initializing to NaN.
#    if f_new != f_new:
#      f_new, g_new = obj_func(x, torch.tensor(1), d)
    ls_func_evals = 1
#TODO: why don't we scale d by t here, especially since we are normalizing?
    gtd_new = (g_new * d).sum() # Keep as scalar tensor
#    g_new = g_new#
#    gtd_new = gtd_new#
    success = False
    is_nan = False

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
#    g_prev = g_prev.to(direction_device)
    done = False
    ls_iter = 0 # Initialize ls_iter

#    t_best = t
    t = torch.tensor(t) # Ensure t is a tensor before the loop
    t_best = t
    device = gtd.device
    f_best = torch.tensor(f, device=device)
    g_best = g
    gtd_best = gtd
#Relaxed Wolfe initialization
    if f_new < f_best  and done != True and f_new == f_new and f_new <= (f - abs(c1 * t * gtd)):
#        if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best:  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
#NOTE: prevent using the first iteration, so that we know we fulfilled the armijo condition. Theres a cleaner way to do this
      success = True
      stall_wolfe = 0
      t_best = t
      f_best = torch.tensor(f_new, device=device)
#TODO this should be a non-blocking offload
      g_best = g_new.to(direction_device)
      gtd_best = gtd_new

#    g = g.to(direction_device)
    gc.collect()
    ls_iter=0
    stall_wolfe=0

#TODO something is broken in insta wolfe. Check the initialization. The same step size doesnt throw when not insta-wolfing
    while ls_iter < max_ls:
#TODO: we can calculate the delta here for insta wolfes and adjust t by the difference, essentially measuring the drift of the interpolation to see if its shifting left or right to try to stay in the min as long as possible over time
#TODO: e.g.: if wolfe is increasing shift up t, if armijo is increasing, shift down t. We may be able to formulate this as a linear equation or a ratio
        # check conditions #TODO: <= for ward condition should be < and just allow first iteration to not check ward condition #TODO: this can increase loss if f_best is greater than current loss (last iteration loss)
#        if f_new > (f - abs(c1 * t * gtd))  or (f_new != f_new and is_nan == True): # or f_new >= f_prev: #NOTE: Ward condition
#        if f_new > (f + c1 * t * gtd)  or (f_new != f_new and is_nan == True)    :
#        print("Ward condition: " + str((gtd_new + gtd_prev)/(f_new - f_prev) ))
#        print("gtd delta: " + str(gtd_new - gtd_prev))
#        if (gtd_new + gtd_prev) / (f_new - f_prev)< c1 and (gtd_new - gtd_prev != 0 ) or f_new != f_new:
        if f_new > (f+ c1 * t * gtd)  or (f_new != f_new and is_nan == True):  #or f_new >= f_prev: #NOTE: Ward condition
#Insta-Wolfe reset to avoid jumbo bracket zoom phase
#            if t_prev == 0:
#              t = 1
#              continue
#            else:
              bracket = [t_prev, t]
              bracket_f = [f_prev, f_new]
  #            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
              bracket_g = [g_prev, g_new]
              bracket_gtd = [gtd_prev, gtd_new]
              break

##TODO: <= for ward condition should be < and just allow first iteration to not check ward condition
        if abs(gtd_new) <= abs(-c2 * gtd) and f_new < f :
            bracket = [t]  #type: ignore[list-item]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            success = True
            t_best = t
            f_best = torch.tensor(f_new, device=device)
            g_best = g_new.to(direction_device)
#TODO: we got NaN on a fast wolf here (not instant)on ys (loss was good but ys returned a NaN
            print("FAST WOLFE")
            break

#TODO: we can still totally bracket the wrong convexity with this because we are moving the minima bracket each time? trace this out
        if gtd_new >= 0 :
            print("NOT DESCENDING")
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

#TODO: since we reuse the last step size, we should bracket in the direction of the first interpolation direction, and change the corresponding zoom break condition if bracketing down instead of up
#TODO: increase 100 and consider tuning 0.1 further
        min_step = t + capture_min_step * (t - t_prev)#TODO: this can miss, if t+0.01 breaks both armijo and wolfe condition (the interpolation is steep)
        lower_bracket = min(t_prev, t)
        upper_bracket = max(t_prev, t)
        max_step = upper_bracket * capture_max_step
#TODO: insufficient progress for bracket maybe? set min_step = t and if t doesnt change then break or nudge here, we miss the point on bracketing too

        # interpolate
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
#TODO: insta-NaN handler
        if f_new != f_new:  #Check for NaN
#          t = torch.tensor(1., device=device) # Reset t to 1.0 on the correct device
#          min_step = torch.tensor(0., device=device) # Reset min_step to 0.0 on the correct device
#          t = _cubic_interpolate(
#              t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
#          )
          is_nan = True
        t = torch.tensor(t, device=device) # Ensure t is a tensor on the correct device

        # next step
        t_prev = tmp
        f_prev = f_new
#        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        g_prev = g_new.to(direction_device)
        gtd_prev = gtd_new # type: ignore[assignment] # type: ignore[assignment]
        f_new, g_new = obj_func(t, d)
        ls_func_evals += 1 # Increment func evals after new evaluation
        gtd_new = (g_new * d).sum() # Keep as scalar tensor
#        g_new = g_new#
        ls_iter += 1
        #RELAXED WOLFE CONDITION
#        cur_c2 =  abs(gtd_new) - -gtd  #TODO: inverted case
#        if f_new < f_best  and done != True and f_new == f_new : #
        if f_new < f_best  and done != True and f_new == f_new :
#        if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best:  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
#NOTE: prevent using the first iteration, so that we know we fulfilled the armijo condition. Theres a cleaner way to do this
          success = True
          stall_wolfe = 0
          t_best = t
          f_best = torch.tensor(f_new, device=device)
#TODO this should be a non-blocking offload
          g_best = g_new.to(direction_device)
          gtd_best = gtd_new

    # reached max number of iterations?
    if ls_iter == max_ls:
#TODO: this is actually better, big zoom if we are out of iterations.
#        bracket = [0, t]
#        bracket_f = [f, f_new]
#        bracket_g = [g, g_new]
#        bracket_gtd = [gtd, gtd_new]
        bracket = [t_prev, t]
        bracket_f = [f_prev, f_new]
        bracket_g = [g_prev, g_new]
        bracket_gtd = [gtd_prev, gtd_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.
    # exact point satisfying the criteria
    # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.

    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
#    while not done and ls_iter < max_ls:
    # zoom phase: we now have a point satisfying the criteria, or # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.

    #NOTE: we wait for bracket to collapse, we dont use max linesearch here, if it takes too long turn the bracket hyperparameters up.
    while not done  and ls_iter < max_ls and not is_nan:
#        if len(bracket) < 2: # Check if bracket has at least 2 elements
#            print("WOLFE PACK")
#            return success, f_best, g_best, t_best, ls_func_evals

            # line-search bracket is so small
            #TODO: extract stall_wolfe hyperparameter
            #        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change or ls_iter >= max_ls or stall_wolfe >= 4:   # type: ignore[possibly-undefined]
        print("zooming..")
        if abs(bracket[1] - bracket[0])  < tolerance_change  : # or stall_wolfe >= 5 :
#TODO: getting negative ys here with stall wolfe printout
           print("WOLFE PACK")
           return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals
         #TODO: return the wolfe pack here
       #            break

        t_prev = t
        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0], # type: ignore[possibly-undefined]
            bracket_gtd[0],
            bracket[1],
            bracket_f[1], # type: ignore[possibly-undefined]
            bracket_gtd[1],
        )
        t = torch.tensor(t, device=device) # Ensure t is a tensor on the correct device
        # insta-NaN handler
#TODO: bracket collapses when we get NaN. Ensure we reset step size accordingly.
#TODO: were jumping the border here
        if f_new != f_new:
#TODO: test this since 1 can cause problems since its the same as the gradient for initialization causing inf delta
          t = torch.tensor(1.)
          is_nan = True
#TODO: need to revaluate here.
#        bracket_gtd[1]#,
#        bracket_gtd[0]#,  # type: ignore[possibly-undefined]

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        #  TODO: This needs to be set based on how large our brackets are. We miss the point with these literal parameters when we arent zooming a large domain.
        eps = bracket_shift * (max(bracket) - min(bracket))
#        eps = tolerance_change * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 1/3 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    displacement = max(bracket) - eps
                    t = torch.tensor(t - bracket_shove*(t - displacement), device=device)
                    print("punt", end = " ")
                else:
                    displacement = min(bracket) + eps
                    t = torch.tensor(t + bracket_shove*(displacement - t), device=device)
                    print("punt", end = " ")
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(t, d) # Single evaluation
        ls_func_evals += 1 # Increment func evals
        gtd_prev = gtd_new
        gtd_new = (g_new * d).sum() # Keep as scalar tensor

#TODO: something like this. we need to move t by the amount that linearly should bring it into c3 notch if we are over converged.
#TODO: also this should happen before evaluation and probably be an if condition against push/shove routine
#TODO: prompt it..
#        if abs(gtd_new) <= -0.4 * gtd:
#          gtd_delta = gtd_new - gtd_prev
#          t_delta = t - t_prev
#TODO: we also need the distance to the notch.
#          t = t*t_delta/gtd_delta

        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?

#        if f_new < f_best  and f_new == f_new:
#        if f_new < f_best  and done != True and f_new == f_new :
#            if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best:  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
##          print("---GOT NEW WOLFE PACK---")
#          success = True
#          stall_wolfe = 0
#          t_best = t
#          f_best = torch.tensor(f_new, device=device)
#          g_best = g_new.to(direction_device)
#          gtd_best = gtd_new
        print("Ward condition: " + str((gtd_new + gtd_prev)/(f_new - f_prev) ))

#        if (f_new - f_prev) / (gtd_new + gtd_prev) < c1  and abs(gtd_new - gtd_prev) != 0 or f_new >= bracket_f[low_pos] or f_new != f_new:
        if f_new > (f + c1 * t * gtd)  or f_new >= bracket_f[low_pos] or f_new != f_new: #or f_new > f_best: #NOTE: Ward condition#NOTE: PREV SETTING
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0) # type: ignore[possibly-undefined]
        else:
            if abs(gtd_new) <= abs(-c2 * gtd) and f_new < f_best : #NOTE: Ward condition #TODO: Ward condition should be < not <=, it should be based on < and if gtd is under a threshold such that we cant get a gtd delta
                # Wolfe conditions satisfied
                print("STRONG WOLFE")
                success = True
                done = True
#TODO: clean up the line search a bit we have a lot of redundancies now and artifacts from deprecated features
                t_best = t
                f_best = torch.tensor(f_new, device=device)
                g_best = g_new.to(direction_device)
            elif gtd_new * (bracket[high_pos] - bracket[low_pos])>= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            #RELAXED WOLFE CONDITION
    #        cur_c1 = (f + t*gtd) - f_new
#            cur_c2 =  abs(gtd_new) - -gtd  #TODO: inverted case
    #NOTE: relaxed wolfe condition. If we fail to find a wolfe we go for best curvature to condition the Hessian.
#            if f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity. #TODO redundant NaN check
#TODO redundant NaN check
#TODO: we have a problem with ys here +gtd - -gtd_new == ys < 0
            if f_new < f_best and f_new == f_new:
##            if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best:  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
#    #          print("---GOT NEW WOLFE PACK---")
              success = True
              stall_wolfe = 0
              t_best = t
              f_best = torch.tensor(f_new, device=device)
              g_best = g_new.to(direction_device)

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
#            bracket_g[low_pos] = g_new.clone() # type: ignore[possibly-undefined]
            bracket_g[low_pos] = g_new
# type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new
        stall_wolfe += 1
        if stall_wolfe >= 5:
          print("STALL WOLFE")
#TODO: there is still a potential but unlikely bug here where we need to account for if loss isnt reduced. Likely we should consider the Armijo in relaxed wolfe
        if ls_iter >= max_ls and done != True and success == True: # Return Wolfe pack if max ls reached in zoom phase
          print("WOLFE PACK MAX LS")
          return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals

    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
#    return success, f_new, g_new.to(optimizer_device), t, ls_func_evals
    return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals

class FBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    # Memory allocation strategies:
    # - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (reduces fragmentation for variable size tensors)
    # - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:<value> (experiment with segment split size, less direct impact on fragmentation)
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-8,
        tolerance_change: float = 1e-8,
        history_size: int = 2,
        c1: float = 1e-3,
        c2: float = 0.25,
        line_search_fn: Optional[str] = None,
        bracket_shift: float =(1/3),
        bracket_shove: float =(1/3),
        capture_min_step: float =1.,
        capture_max_step: float =100,
        radius_alpha: float = 0,
        direction_device: str = 'cpu',
        optimizer_device: str = 'cuda',
        norm: float = 1.0,
        y_norm: Optional[float] = None,
        rho_rewind: int = 10,
        orthogonality: float = 1e-2,
        max_ls: int = 10
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            c1=c1,
            c2=c2,
            line_search_fn=line_search_fn,
            bracket_shift=bracket_shift,
            bracket_shove=bracket_shove,
            capture_min_step=capture_min_step,
            capture_max_step=capture_max_step,
            radius_alpha=radius_alpha,
            direction_device=direction_device,
            optimizer_device=optimizer_device,
            norm=norm,
            y_norm=y_norm,
            rho_rewind=rho_rewind,
            orthogonality=orthogonality,
            max_ls=max_ls
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "FBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self.radius_alpha = radius_alpha
        self.direction_device = direction_device
        self.optimizer_device = optimizer_device
        self.max_ls= max_ls
        self.max_iter = max_iter
        self.t = 1

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

#TODO: allow this to be distributive, such that we can have multiple nodes generate grads and keep them on their respective devices for all of fbfgs.
    def _gather_flat_grad(self):
        # Perform gradient clipping before gathering
#        torch.nn.utils.clip_grad_norm_(
#            self._params, 
#            max_norm=0.5,
#            norm_type=2
#        )

        views = []
        for p in self._params: # Iterate over parameters
            grad_device = p.device # Get the device of the gradient (e.g., cuda:1)
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.view(-1) # Move sparse grad to direction_device
            else:
                view = p.grad.view(-1) # Move dense grad to direction_device
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view.to(self.optimizer_device))
        grad = torch.cat(views, 0)

#        grad_norm = torch.linalg.vector_norm(grad, ord=2.)
#        grad = grad.div_(grad_norm)
#        grad_max = grad.max()
#TODO: this is a good idea but I would prefer a more functional and elegant way to handle rollover since it can in theory occur throughout the algorithm and pytorch doesnt solve this elegantly (which it should).
        grad = torch.nan_to_num(grad, nan=0.0) #, posinf=finfo.max, neginf=finfo.min)
        return grad

    def _add_grad(self, step_size, update, limit_offset: int = -1) -> int:
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()

            current_param_end_offset = offset + numel
            slice_end = min(current_param_end_offset, limit_offset if limit_offset != -1 else current_param_end_offset)
            slice_numel = slice_end - offset

            if slice_numel <= 0:
                offset += numel
                continue

            # Apply update to parameter p_view
            if isinstance(update, SparseFlatTensor):
                # For sparse updates, use the dedicated function
                # step_size acts as the alpha parameter here
                # Note: _add_sparse_dense_alpha modifies the dense tensor passed to it.
                # We pass p.view(-1) which is the flattened parameter.
                # --- Key Change: Pass the current offset ---
                SparseFlatTensor._add_sparse_dense_alpha(update, p.view(-1), alpha=step_size, offset=offset)
            else:
                # For dense updates, use in-place add_
                view = update[offset : offset + slice_numel].to(p.device)
                # Get the slice of the parameter tensor directly
                p_slice = p.view(-1)[0:slice_numel]

                # Use in-place add_ and remove NaN check/clone logic
                # This directly modifies p_slice, which is a view into p.view(-1)
                p_slice.add_(view.view_as(p_slice), alpha=step_size)

            offset += numel
        for p in self._params:
          p = torch.nan_to_num(p, nan=0.0) #, posinf=finfo.max, neginf=finfo.min)
        return self._numel() # Return total numel if no NaN

    def _directional_evaluate(self, closure, t, d):
        # Save current parameters to CPU
        original_params_cpu = [p.detach().clone().cpu() for p in self._params]
#        original_params_cpu = [p.pin_memory() for p in original_params_cpu]
        # Apply step: x_new = x_old + t * d
        offset = 0
        for p in self._params:
            numel = p.numel()
            if torch.is_complex(p):
                p_view = torch.view_as_real(p).view(-1)
            else:
                p_view = p.view(-1)

            # Updated callsite: use the new _add_sparse_dense_alpha function
            if isinstance(d, SparseFlatTensor):
                # --- Key Change: Pass the current offset ---
                SparseFlatTensor._add_sparse_dense_alpha(d, p_view, alpha=t, offset=offset)
            else:
                # Assume d is a dense tensor
                p_view.add_(d[offset : offset + numel].to(p.device), alpha=t)
            offset += numel

        loss = float(closure())
        flat_grad = self._gather_flat_grad()

        # Restore original parameters from CPU
        for p, original_p_cpu in zip(self._params, original_params_cpu): # type: ignore[possibly-undefined]
            p.copy_(original_p_cpu.to(p.device))

        return loss, flat_grad

    def sparse_direction_approximate(self, old_stps: list[SparseFlatTensor], old_dirs: list[SparseFlatTensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, optimizer_device: str, t: float, radius_alpha: float, norm: float, y_norm: float, ls_failed: bool, orthogonality: float, n_iter: int) -> Tensor:
        torch.cuda.synchronize() # Ensure all previous CUDA operations are complete, especially non-blocking transfers to calculation device
        num_old = len(old_dirs)
        hit_miss = str("")
        # Similarity threshold

        q = flat_grad.to(torch.float32).to(optimizer_device).neg()
        total_norm = torch.linalg.vector_norm(q, ord=2.).to(torch.float32).to(optimizer_device)
        total_norm = total_norm / self.radius_alpha
        print("q max value: " + str(q.max()))
        if total_norm == float('inf') or total_norm == 0:  
          print("pre-direction l2 norm returned inf")
        q = q.div_(total_norm)

        al = torch.empty(num_old, dtype=q.dtype, device=optimizer_device)
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=optimizer_device)

        if num_old > 0:
            # Prefetch the first element for the backward loop
#TODO: we may want to allow a buffer for latency/throughput tuning for various PCIE/accelerator configurations.
            next_sparse_dir_prefetch: SparseFlatTensor = old_dirs[num_old - 1].to(torch.device(optimizer_device), non_blocking=True)
            next_sparse_stp_prefetch: SparseFlatTensor = old_stps[num_old - 1].to(torch.device(optimizer_device), non_blocking=True)

            for i in range(num_old - 1, -1, -1):
                torch.cuda.synchronize() # Ensure current_sparse_dir is ready
#                current_sparse_dir_val = torch.jit.annotate(SparseFlatTensor, next_sparse_dir_prefetch) # Get the prefetched tensor
                current_sparse_stp_val = torch.jit.annotate(SparseFlatTensor, next_sparse_stp_prefetch) # Get the prefetched tensor
                current_sparse_dir_val = torch.jit.annotate(SparseFlatTensor, next_sparse_dir_prefetch) # Get the prefetched tensor

#TODO: should we be async transfering Rho? are we broadcast multing on the CPU?
                if i > 0: #(Supercharger)
                    # Initiate async pinning for next iteration while current operations run
                    if i == num_old - 1:  # First iteration - submit async pinning
                        src_dir_next = old_dirs[i - 1]
                        src_stp_next = old_stps[i - 1]
                        if src_dir_next.values.device.type == 'cpu':
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                # Submit pinning ops to thread pool
                                dir_future = executor.submit(lambda: src_dir_next.pin_memory())
                                stp_future = executor.submit(lambda: src_stp_next.pin_memory())
                                # Store futures for next iteration
                                self.dir_pinned, self.stp_pinned = dir_future, stp_future
                    else:  # Subsequent iterations - use previously pinned memory
                        # Ensure futures exist and aren't missing
                        assert hasattr(self, 'dir_pinned') and hasattr(self, 'stp_pinned'), "Pinned futures missing!"
                        src_dir_pinned = self.dir_pinned.result()
                        src_stp_pinned = self.stp_pinned.result()
                        # Non-blocking GPU transfer 
                        next_sparse_dir_prefetch = src_dir_pinned.to(torch.device(optimizer_device), non_blocking=True)
                        next_sparse_stp_prefetch = src_stp_pinned.to(torch.device(optimizer_device), non_blocking=True)

                    # Submit pinning for next-next iteration (except last)
                    if i > 1:
                        src_dir_next = old_dirs[i - 2]
                        src_stp_next = old_stps[i - 2]
                        if src_dir_next.values.device.type == 'cpu':
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                # Submit pinning ops in background
                                dir_future = executor.submit(lambda: src_dir_next.pin_memory())
                                stp_future = executor.submit(lambda: src_stp_next.pin_memory())
                                # Store futures for next iteration
                                self.dir_pinned, self.stp_pinned = dir_future, stp_future

                # Create a new SparseFlatTensor with internal values cast to float32
#                sparse_dir_i = SparseFlatTensor(
#                    current_sparse_dir_val.starts, current_sparse_dir_val.ends, current_sparse_dir_val.values.to(dtype=torch.float32),
#                    current_sparse_dir_val.total_size, current_sparse_dir_val.unit_indices, current_sparse_dir_val.unit_values.to(dtype=torch.float32)
                sparse_stp_i = SparseFlatTensor(
                    current_sparse_stp_val.starts, current_sparse_stp_val.ends, current_sparse_stp_val.values.to(dtype=torch.float32),
                    current_sparse_stp_val.total_size, current_sparse_stp_val.unit_indices, current_sparse_stp_val.unit_values.to(dtype=torch.float32)
                )
                sparse_dir_i = SparseFlatTensor(
                    current_sparse_dir_val.starts, current_sparse_dir_val.ends, current_sparse_dir_val.values.to(dtype=torch.float32),
                    current_sparse_dir_val.total_size, current_sparse_dir_val.unit_indices, current_sparse_dir_val.unit_values.to(dtype=torch.float32)
                )
#                direction_similarity = SparseFlatTensor.sparse_dot_dense(sparse_dir_i, q).item()
#                alpha = SparseFlatTensor.sparse_dot_dense(sparse_stp_i, q).item()

                direction_similarity = SparseFlatTensor.sparse_dot_dense(sparse_dir_i, q).item()
                aligned = (direction_similarity <= orthogonality  and direction_similarity >= -orthogonality)  or i >= num_old - n_iter 
#                aligned = (direction_similarity >= orthogonality  and direction_similarity <= -orthogonality) or i > num_old - n_iter # # or i > num_old - n_iter
                direction_alignment_mask[i] = aligned
                if direction_alignment_mask[i]:
                  alpha = SparseFlatTensor.sparse_dot_dense(sparse_stp_i, q).item()
                  al[i] = alpha * ro[i].item()
                  sparse_old_dir_scaled = SparseFlatTensor(
                      current_sparse_dir_val.starts, current_sparse_dir_val.ends, current_sparse_dir_val.values.to(dtype=torch.float32),
                      current_sparse_dir_val.total_size, current_sparse_dir_val.unit_indices, current_sparse_dir_val.unit_values.to(dtype=torch.float32)
                  ) * ((-al[i]))
#TODO: to perform orthogonalization with the direction corrected first loop we just have to take sparse_old_dir_scaled without Rho as the direction_similarity, then mul rho for adding to q
                  q = SparseFlatTensor._add_sparse_dense(sparse_old_dir_scaled, q)
                  q_norm = torch.linalg.vector_norm(q, ord=2.) / radius_alpha
                  if q_norm > 0:
                      q = q.div_(q_norm)
                  hit_miss = hit_miss + str("| ")
                else:
                  hit_miss = hit_miss + str("_ ")

        print("q max value: " + str(q.max()))
#        q_norm = torch.linalg.vector_norm(q, ord=2.)
#        if q_norm > 0:
#            q = q.div_(q_norm)
#        d = q.mul(H_diag.to(torch.float32))
        d = q
        del q

        if num_old > 0:
            aligned_indices = torch.nonzero(direction_alignment_mask).squeeze(1)
            if aligned_indices.numel() > 0:
                # Prefetch the first aligned element 
                first_aligned_idx = aligned_indices[0]
                # Initiate async pinning for first aligned element if needed
                if old_dirs[first_aligned_idx].values.device.type == 'cpu':
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        dir_future = executor.submit(lambda: old_dirs[first_aligned_idx].pin_memory())
                        stp_future = executor.submit(lambda: old_stps[first_aligned_idx].pin_memory())
                        next_dir_pinned = dir_future.result()
                        next_stp_pinned = stp_future.result()
                    next_old_dir_prefetch_fwd = next_dir_pinned.to(torch.device(optimizer_device), non_blocking=True)
                    next_old_stp_prefetch_fwd = next_stp_pinned.to(torch.device(optimizer_device), non_blocking=True)
                else:
                    next_old_dir_prefetch_fwd = old_dirs[first_aligned_idx].to(torch.device(optimizer_device), non_blocking=True)
                    next_old_stp_prefetch_fwd = old_stps[first_aligned_idx].to(torch.device(optimizer_device), non_blocking=True)

                for k in range(aligned_indices.numel()):
                    i = aligned_indices[k] # Get the original index
                    torch.cuda.synchronize() # Ensure current_old_dir and current_old_stp are ready
                    current_old_dir_val = torch.jit.annotate(SparseFlatTensor, next_old_dir_prefetch_fwd)
                    current_old_stp_val = torch.jit.annotate(SparseFlatTensor, next_old_stp_prefetch_fwd)
                    
                    if k < aligned_indices.numel() - 1:
                        next_aligned_idx = aligned_indices[k + 1]
                            
                        # Initiate async pinning for next iteration while current operations run
                        src_dir_next = old_dirs[next_aligned_idx]
                        src_stp_next = old_stps[next_aligned_idx]
                        
                        if src_dir_next.values.device.type == 'cpu':
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                # Submit pinning ops to thread pool
                                dir_future = executor.submit(lambda: src_dir_next.pin_memory())
                                stp_future = executor.submit(lambda: src_stp_next.pin_memory())
                                # Store futures for next iteration
                                self.fwd_dir_pinned, self.fwd_stp_pinned = dir_future, stp_future
                                
                        # Non-blocking transfer with pinned memory
                        assert hasattr(self, 'fwd_dir_pinned') and hasattr(self, 'fwd_stp_pinned'), "Pinned futures missing!"
                        next_dir_pinned = self.fwd_dir_pinned.result()
                        next_stp_pinned = self.fwd_stp_pinned.result()
                        next_old_dir_prefetch_fwd = next_dir_pinned.to(torch.device(optimizer_device), non_blocking=True)
                        next_old_stp_prefetch_fwd = next_stp_pinned.to(torch.device(optimizer_device), non_blocking=True)

                    old_dir_for_dense = SparseFlatTensor(
                        current_old_dir_val.starts, current_old_dir_val.ends, current_old_dir_val.values.to(dtype=torch.float32), # type: ignore[arg-type]
                        current_old_dir_val.total_size, current_old_dir_val.unit_indices, current_old_dir_val.unit_values.to(dtype=torch.float32)
                    )
                    dot_product_val = SparseFlatTensor.sparse_dot_dense(old_dir_for_dense, d)
                    alpha_val = al[i] - dot_product_val * ro[i].item()
                    sparse_old_stp_scaled = SparseFlatTensor(
                        current_old_stp_val.starts, current_old_stp_val.ends, current_old_stp_val.values.to(dtype=torch.float32),
                        current_old_stp_val.total_size, current_old_stp_val.unit_indices, current_old_stp_val.unit_values.to(dtype=torch.float32)
                    ) * (alpha_val)
                    d = SparseFlatTensor._add_sparse_dense(sparse_old_stp_scaled, d)#TODO: test the in place operation to avoid a likely non-DCE'd clone

        print(f"sparse_direction_approximate hit_miss: {hit_miss}")
        total_norm = torch.linalg.vector_norm(d, ord=norm).to(torch.float16).to(optimizer_device)
        scaled_norm = total_norm / radius_alpha
        print(f"max value pre-norm direction: {d.max()}, scaled_norm: {scaled_norm}")
        d = d.to(torch.float16).div_(scaled_norm)

        total_norm = torch.linalg.vector_norm(d, ord=2.).to(torch.float16).to(optimizer_device)
        total_norm = total_norm / self.radius_alpha
        d = d.div_(total_norm)

        d = torch.nan_to_num(d, nan=0.0) #, posinf=finfo.max, neginf=finfo.min)
#        d = d.mul_(total_norm)
        return d

    @torch.jit.script
    def dense_direction_approximate(old_stps: list[Tensor], old_dirs: list[Tensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, optimizer_device: str, t: float, radius_alpha: float, norm: float) -> Tensor:
        torch.cuda.synchronize() # Ensure all previous CUDA operations are complete, especially non-blocking transfers to calculation device
        num_old = len(old_dirs)
        hit_miss = str("")
        similarity = 0.
#TODO: underflow also this should be better formulated and we should try to avoid another hyperparam but arbitrary literals are worse than hyperparams
        if t < 1:
          similarity = similarity/t
        q = flat_grad.neg().to(flat_grad.device)
        total_norm = torch.linalg.vector_norm(q, ord=2.).to(q.device) # Move total_norm to q's device
        q = q.div_(total_norm)
#        mask = torch.logical_and(q > -radius_alpha, q < radius_alpha) #TODO: extract to sub_variance hyperparameter

        al = torch.empty(num_old, dtype=q.dtype, device=q.device) # Initialize al as tensor on q's device
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=q.device) # Initialize mask on q's device

        if num_old > 0:
            # Prefetch the first element for the backward loop
            next_old_dir_prefetch_bwd: Tensor = old_dirs[num_old - 1].to(q.device, non_blocking=True)

            for i in range(num_old - 1, -1, -1):
                torch.cuda.synchronize() # Ensure current_old_dir is ready for consumption
                current_old_dir_val = torch.jit.annotate(Tensor, next_old_dir_prefetch_bwd)
                if i > 0:
                    next_old_dir_prefetch_bwd = old_dirs[i - 1].to(q.device, non_blocking=True)
                direction_similarity = (current_old_dir_val * q).sum().item() # Use current_old_dir_val
                aligned = direction_similarity >= similarity or direction_similarity <= -similarity
                direction_alignment_mask[i] = aligned # Store alignment for current index
                if direction_alignment_mask[i]:
                  al[i] = direction_similarity * ro[i].item()
                  q = q + (current_old_dir_val * ((-al[i])))
                  hit_miss = hit_miss + str("| ")
    # TODO: prevent over-alignment to keep the direction multipathed?
    # Prevent over-alignment by considering the expansion of near-orthogonal entries
    #              if direction_similarity < 1 and direction_similarity > -1:
    ##TODO: over-alignment hyperparameter (last one I swear this one is really necessary)
    #                similarity += 0.1*similarity*(1 - abs(direction_similarity)) #TODO: we assume worst case which is variance has doubled ?  We can calculate this based on the alignment. the less aligned the more variance in the solution.
    #              else:
    #                similarity += 5e-8 #TODO: a better way to prevent PowerPoints
    #              similarity += 0.1*similarity
                else:
                  hit_miss = hit_miss + str("_ ")

#        d = torch.nan_to_num(q.mul(H_diag.to(optimizer_device)), nan=0.0, posinf=0.0, neginf=0.0) # Ensure H_diag is on optimizer_device
        d = q.mul(H_diag.to(optimizer_device))
        be_i = torch.empty_like(d, dtype=q.dtype, device=d.device) # Preallocate be_i for second loop on d's device
        del q

        if num_old > 0:
            # Prefetch the first elements for the forward loop
            next_old_dir_prefetch_fwd: Tensor = old_dirs[0].to(d.device, non_blocking=True) # Prefetch to d's device
            next_old_stp_prefetch_fwd: Tensor = old_stps[0].to(d.device, non_blocking=True) # Prefetch to d's device

            #TODO: vectorize alignment mask here since its immutable
            for i in range(num_old):
                torch.cuda.synchronize() # Ensure current_old_dir and current_old_stp are ready
                current_old_dir_val = torch.jit.annotate(Tensor, next_old_dir_prefetch_fwd)
                current_old_stp_val = torch.jit.annotate(Tensor, next_old_stp_prefetch_fwd)
                if i < num_old - 1:
                    next_old_dir_prefetch_fwd = old_dirs[i + 1].to(d.device, non_blocking=True)
                    next_old_stp_prefetch_fwd = old_stps[i + 1].to(d.device, non_blocking=True) # Prefetch to d's device

                if direction_alignment_mask[i]: # Check alignment for current index
                  be_i.copy_((current_old_dir_val * d)) # Use current_old_dir_val
                  alpha_val = al[i] - be_i.sum() * ro[i].item() # Use al[i] and ro[i]
                  d = d + (current_old_stp_val * (alpha_val)) # Use current_old_stp_val

#        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        print(hit_miss)
        total_norm = torch.linalg.vector_norm(d, ord=norm).to(optimizer_device)
        scaled_norm = total_norm / radius_alpha
        d = d.div_(scaled_norm)
        return d

    @torch.no_grad()
    def step(self, closure):
      """Perform a single optimization step.

      Args:
          closure (Callable): A closure that reevaluates the model
              and returns the loss.
      """
      assert len(self.param_groups) == 1

      # Make sure the closure is always called with grad enabled
      closure = torch.enable_grad()(closure)

      group = self.param_groups[0]
      lr = group["lr"]
      max_iter = group["max_iter"]
      max_eval = group["max_eval"]
      tolerance_grad = group["tolerance_grad"]
      tolerance_change = group["tolerance_change"]
      line_search_fn = group["line_search_fn"]
      history_size = group["history_size"]
      c1 = group["c1"]
      c2 = group["c2"]
      bracket_shift=group["bracket_shift"]
      bracket_shove=group["bracket_shove"]
      capture_min_step=group["capture_min_step"]
      capture_max_step=group["capture_max_step"]
      norm = group["norm"]
      y_norm = group["y_norm"]
      rho_rewind = group["rho_rewind"]
      orthogonality = group["orthogonality"]

      # NOTE: FBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      state = self.state[self._params[0]]

      # evaluate initial f(x) and df/dx
      orig_loss = closure()
        #TODO: this should probably be direction_evaluate so we include the radius_alphaping factor
      loss = float(orig_loss)
      current_evals = 1
#      state["func_evals"] += 1
      al = []

#TODO: put old_dirs, steps and ro on self.direction_device. Perform the direction calculation as efficiently as possible with this constraint so we can use main memory for history size
      # tensors cached in state (for tracing) - ensure they are always lists
      old_dirs = state.get("old_dirs", [])
      old_stps = state.get("old_stps", [])
      ro = state.get("ro", [])
      d = state.get("d", None) # d can be None initially

      # Initialize prev_flat_grad, flat_grad, and H_diag
      prev_flat_grad = state.get("prev_flat_grad", None)
      flat_grad = state.get("flat_grad", None)
      if "H_diag" in state:
        H_diag = state.get("H_diag")
      else:
        H_diag = 1
        H_diag = torch.tensor(H_diag)
      H_diag = 1
      H_diag = torch.tensor(H_diag)
#      flat_grad = None
#      prev_flat_grad = None # Initialize prev_flat_grad to None
#
      n_iter = 0
      new_ys_x = 0
#      d = flat_grad.neg() # Initialize d on direction_device
      first_param = next(self.param_groups[0]['params'].__iter__())
      t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)
      ls_failed = False
      ls_failed = state.get("ls_failed", False) # Retrieve ls_failed state
#TODO: we arent storing the last iteration in history. Consider reworking the last iteration logic for step function
#      while n_iter < max_iter:
      while True:
          torch.cuda.empty_cache() # Clear cache before direction calculation
          # keep track of nb of iterations
          n_iter += 1
          print("iteration: " + str(n_iter))
          print("[CRAM]")

          ############################################################
          # compute gradient descent direction
          ############################################################
          # If this is the first iteration or history was reset
#TODO: add a special condition such that if num iters is 1 we start with the direction otherwise we do the gradient.
          if  n_iter== 1 or prev_flat_grad is None:#TODO: dont need to check prev_flat_grad here since we dont use the needle anymore
#          if prev_flat_grad is None:
              restart = False # Flag for restart
#TODO: use the proper flat_grad (the l1 instead of l2) here since we don't calculate direction first
              print("RESET (n_iter=1 or prev_flat_grad is None)")
              flat_grad = self._gather_flat_grad().to(self.optimizer_device)
#TODO: clip_grad_norm by the l1 norm for a max norm of 1e9 (if needed)
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9)
              if flat_grad.abs().max() <= tolerance_grad: #TODO: check if this is even possible given normalization.
                return orig_loss
              H_diag = 1
              H_diag = torch.tensor(H_diag, device=self.optimizer_device) # Ensure H_diag is on optimizer_device
              if len(old_dirs) > 0  and n_iter > 1:
                d = self.sparse_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, optimizer_device=self.optimizer_device, t=t, radius_alpha=self.radius_alpha, norm=norm, y_norm = y_norm, ls_failed=ls_failed, orthogonality=orthogonality, n_iter=new_ys_x)
              else:
                d = self.sparse_direction_approximate([], [], [], flat_grad, H_diag, optimizer_device=self.optimizer_device, t=t, radius_alpha=self.radius_alpha, norm=norm, y_norm = y_norm, ls_failed=ls_failed, orthogonality=orthogonality, n_iter=n_iter)
#                  d = self._gather_flat_grad().neg().to(self.optimizer_device) # Ensure d is on optimizer_device
#                  #TODO: should we also do norm float("inf") here to match direction S?
#    #              total_norm = torch.linalg.vector_norm(d, ord=2.).to(self.optimizer_device)  
#    #              d = d.div_(total_norm)
#                  d = d.div_(total_norm)
#                  total_norm = torch.linalg.vector_norm(d, ord=norm).to(self.optimizer_device) # Calculate norm on optimizer_device
#                  print("d norm: " + str((total_norm)) )
#                  d = d.div_(total_norm)
#                  d[torch.logical_and(d > -self.radius_alpha,d < self.radius_alpha)] = 0
#                  total_norm = torch.linalg.vector_norm(d, ord=2.).to(self.optimizer_device) # Calculate norm on optimizer_device
#                  print("d norm: " + str((total_norm)) )
#                  d = d.div_(total_norm)
#                  loss, flat_grad = self._directional_evaluate(closure, 1., d)
		#NOTE: end of else
#
#              d = d.to_sparse()
#              d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
#              d = d*total_norm
              gc.collect()
#              print("d elements: " + str((d.values() != 0).sum()) )
#              print("direction elements: " + str((d != 0).sum()) )
          else:
#SELECTION SUBROUTINE
#              norm_flat_grad = flat_grad/total_norm_grad
#
#              prev_norm_flat_grad = prev_flat_grad/total_norm_prev_grad  #Creates new tensor (if not in-place)

              # Calculate y_dense using clone and in-place operations to reduce allocations
#TODO: clip flat_grad and prev_flat_grad here respectively.
              # Apply L2 norm clipping to flat_grad and prev_flat_grad
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9) # Clip flat_grad norm
#              if prev_flat_grad is not None:
#                  torch.nn.utils.clip_grad_norm_(prev_flat_grad, max_norm=1e9)
#TODO: clip flat_grad and prev_flat_grad here respectively (if not already done).
              # prev_flat_grad is moved to optimizer_device for calculation, then immediately deleted
              # to free up memory.
              y_dense = flat_grad.clone() # y_dense is on flat_grad's device (optimizer_device)
              y_dense.sub_(prev_flat_grad.to(self.optimizer_device)) # Ensure prev_flat_grad is on optimizer_device
#              s_dense = (d.mul(t)) # Define s_dense here
              s_sparse = d * t # Define s_sparse here
#              ys = y_dense.dot(s_dense)
              ys = SparseFlatTensor.sparse_dot_dense(s_sparse, y_dense)
              if ys > 0:
                y_squared = y_dense.dot(y_dense)
                H_diag = ys / y_squared  
                del y_squared  
              else:
                H_diag = 1.
                H_diag = torch.tensor(H_diag, device=self.optimizer_device)  
#              if H_diag < 1:
#                H_diag = 1.
#                H_diag = torch.tensor(H_diag, device=self.optimizer_device)  
              # Powell dampening modification
#              delta =1  # existing curvature threshold
##              if 0 < ys < delta:
##                  # Calculate ||s||^2 efficiently from sparse components
##                  ss = (s_sparse.values**2).sum() + (s_sparse.unit_values**2).sum()
##                  if ss > 1e-10:  # avoid division by zero
##                      theta = (delta - ys) / ss
##                      SparseFlatTensor._add_sparse_dense_alpha(s_sparse, y_dense, alpha=theta, offset=0)
##                      ys = SparseFlatTensor.sparse_dot_dense(s_sparse, y_dense)
##                      print(f"\033[94mApplied Powell dampening. New ys: {ys}\033[0m")
##                  else:
##                      print("Skipped Powell dampening due to small ||s||^2")

              original_y_dtype = y_dense.dtype # Store original dtype
#TODO: we may not need y_dense now
#              y_dense = y_dense.clone()
              norm_y_dense = torch.linalg.vector_norm(y_dense, ord=2.) / self.radius_alpha
#TODO: it may be of note that doing selection on the raw y may remove some of the late convergence aspects of the l2 distribution despite being a sample of the l2 distribution. We may need to normalize first (but keep rho on raw) for the y selection
#              norm_y_dense = max(1e-9, norm_y_dense)

              y_dense.div_(norm_y_dense)

              torch.cuda.empty_cache() # Clear cache

#TODO: this kinda throws off selection but also makes it independent of s which may be useful. Note here we may want to do this after the initial selection.
#TODO: this should be done after the div if in place otherwise we dont rescale with the l2 correctly
#              s_mask = s_sparse.get_nonzero_mask()  # Use the new method
#              ys_dense = y_dense.clone()
#              ys_dense[~s_mask] = 0

              #Shotgun noise
#TODO: find the indices of the selected parameters instead of div then multiply since this is lossy in the mantissa
#TODO: this creates an additional tensor make sure this isnt worst case allocation optimizing the tensor ops for memory
#              norm_y = torch.linalg.vector_norm(y_dense, ord=y_norm)
#              y_selection = y_dense.div(norm_y)
#              y_dense_mask = torch.logical_and(y_selection >= -self.radius_alpha, y_selection <= self.radius_alpha)
#              y_dense[y_dense_mask] = 0
#              del y_dense_mask
#              del y_selection
#              y_mask = (y_dense == 0)
#              ys_mask = s_mask & y_mask  # Use & instead of torch.logical_and
#              ys_dense[~ys_mask] = 0
#              print("y dense pre s-mask " + str((y_dense != 0).sum()))
#              print("s mask: " + str((s_mask!=0).sum()))
#              y_dense.add_(ys_dense)
#              print("y dense + s_mask  " + str((y_dense != 0).sum()))
#
#              norm_yf = torch.linalg.vector_norm(y_dense, ord=2.)
#              print("got norm_yf: " +str(norm_yf))
#              y_dense.div_(norm_yf)

#              norm_y_dense = norm_y_dense * yf #if the full vector is the unit distance, this should be proportional
#              del ys_dense
#              del ys_mask
#              del y_mask
#              del s_mask
              torch.cuda.empty_cache()
              gc.collect() # Collect garbage

#TODO consider scale aware thresholding s.t. ys >= 1e-1* s.dot(s).sqr()
#TODO: maybe this should be rho so we dont throw off the Hessian diag
#              yf = self._numel() / (y_dense != 0).sum()
#              ys = ys * yf
#              ys = ys * 1e2
#NOTE: 0.1 is approx 368/30Mill. we may have a better way to formulate this by clipping according to an inverse of yf
#              yf =  (y_dense != 0).sum() / self._numel()
#              if ys <= 0.1 and ys > 0 and t > 1: #NOTE: was 0.1
#                ys = 0.1
#              if ys > 0:
#                y_squared = y_dense.dot(y_dense)
#                H_diag = ys / y_squared  
#                del y_squared  
#              else:
#                H_diag = 1.
#                H_diag = torch.tensor(H_diag, device=self.optimizer_device)  
#TODO: ys = Y*S was here
              print(f"ys: {ys}")
#              # Powell dampening modification
              delta =1   #existing curvature threshold
              if 0 < ys < delta:
#                   Calculate ||s||^2 #efficiently from sparse components
                  ss = (s_sparse.values**2).sum() + (s_sparse.unit_values**2).sum()
                  if ss > 1e-10:   #avoid division by zero
                      theta = (delta - ys) / ss
                      SparseFlatTensor._add_sparse_dense_alpha(s_sparse, y_dense, alpha=theta, offset=0)
                      ys = SparseFlatTensor.sparse_dot_dense(s_sparse, y_dense)
                      print(f"\033[94mApplied Powell dampening. New ys: {ys}\033[0m")
                  else:
                      print("Skipped Powell dampening due to small ||s||^2")

#              if self.radius_alpha != 0:
              y = dense_to_sparse_flat_tensor(y_dense)
#              s = dense_to_sparse_flat_tensor(s_sparse)
#              else:
#                y = y_dense
#                s = s_dense
              print("d-delta elements: " + str((d.to_dense() != 0).sum()) + " total: " + str(d.to_dense().numel()), end=' ')
#              print("S elements: " + str((s_dense != 0).sum()) + " total: " + str(s_dense.numel()), end=' ') # Print S elements
              print("y-delta elements: " + str((y.to_dense() != 0).sum()) + " total: " + str(y.to_dense().numel()), end=' ')
#TODO: this is correct, but maybe there is something more elegant. Possibly reduce based on the mean or the l1/l2 distribution with a hyperparameter. This can be modeled as a outlier distribution problem. We want to maximize Rho so only remove what we need to stabilize the direction-- how we quantify this is TODO
#TODO: this is arguably better than similarity. I wonder if recency matters, such as remove the oldest entries of large Rho (but more elegant)
#TODO: maybe we can even have a weighted pop for the sliding window that considers both the recency and magnitude of the Rho entries? This is all observations on something that needs to be fundamentally quantified.
              if (ys >= 1e-3  ) :#TODO: is this Kosher?
                if self.direction_device != 'cpu' and torch.cuda.is_available():
                  try:
                    cuda_memory_allocated = torch.cuda.memory_allocated(device=self.direction_device) / 1000000000
                    print(f"CUDA memory allocated: {cuda_memory_allocated} GB, history_size: {history_size} GB") # Debug print
                    while cuda_memory_allocated >= history_size:#TODO: history size is the amount of memory available from the device
                        cuda_memory_allocated = torch.cuda.memory_allocated(device=self.direction_device) / 1000000000
                        # shift  history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                  except Exception as e:
                    print(f"CUDA memory check failed: {e}.  Falling back to psutil.")
                elif self.direction_device == 'cpu':
                  try:
                    cpu_ram_available = psutil.virtual_memory().available / (1024**3) # Available RAM in GB
                    print(f"CPU RAM available: {cpu_ram_available} GB, history_size: {history_size} GB") # Debug print
                    while cpu_ram_available <= history_size: # If available RAM is less than history_size and history is not empty
                        if not old_dirs: # Prevent popping from an empty list
                            print("  History is empty, stopping CPU memory-based popping.")
                            break

                        old_available_ram_before_pop = cpu_ram_available # Capture RAM before pop

                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                        gc.collect()

                        # Poll until memory changes or history becomes empty
                        while True:
                            cpu_ram_available = psutil.virtual_memory().available / (1024**3)
                            if cpu_ram_available != old_available_ram_before_pop or not old_dirs:
                                break # Memory changed or history is empty, exit polling
                            time.sleep(0.01) # Small sleep to prevent busy-waiting
                  except Exception as e:
                    print(f"CPU RAM check failed: {e}. Falling back to default memory management.")
                print(f"L-BFGS history popped. History size reduced to: {len(old_dirs)}")
                torch.cuda.empty_cache() # Clear cache before history update
                # Store new direction/step
                old_dirs.append(y.to(self.direction_device, non_blocking=False, pin_memory=False)) # Store y as SparseFlatTensor
                old_stps.append(s_sparse.to(self.direction_device, non_blocking=False, pin_memory=False)) # Store s as SparseFlatTensor #TODO: pinme
# TODO: if we fixed the memory error, reintroduce pinned memory here
                ro.append(torch.tensor([(1. / ys)])) #TODO: pinme
#TODO: we have a problem with ys here +gtd - -gtd_new == ys < 0
                new_ys_x = new_ys_x + 1
              else:
                ls_failed = True
                state["ls_failed"] = True
              if n_iter > max_iter or loss == 0:
                state["old_stps"] = old_stps
                state["ro"] = ro
                state["old_dirs"] = old_dirs
                break
              if flat_grad.abs().max() <= tolerance_grad: #TODO: check if this is even possible given normalization.
                state["old_stps"] = old_stps
                state["ro"] = ro
                state["old_dirs"] = old_dirs
                return orig_loss
              # Update scale of initial Hessian approximation
#              if ys > 0:
#                y_squared = y_dense.dot(y_dense)
#                H_diag = ys / y_squared  
#                del y_squared 
#              else:
#                H_diag = 1.
#                H_diag = torch.tensor(H_diag, device=self.optimizer_device)  
#              H_diag = ys #TODO: just 1?
              gc.collect()

              y = y #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              ys = ys #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.

              del y # Delete y
#              del s

              # compute the approximate (L-BFGS) inverse Hessian
              # multiplied by the gradient
              num_old = len(old_dirs)

              flat_grad = self._gather_flat_grad()
#TODO: may need to try this again? the hessian doesn't pertain as much given that the next direction is likely orthogonal to the last.
#TODO: it might make sense to divide by the history size so we keep curvature normalized to prevent explosions in direction approx.
#              H_diag = 1
#              H_diag = torch.tensor(H_diag)
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9) # Clip gradient norm
#              if self.radius_alpha == 0: # Check if radius_alphaping is disabled
#                d = self.dense_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, radius_alpha=self.radius_alpha, norm=norm)
#              else:
              d = self.sparse_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, optimizer_device=self.optimizer_device, t=t, radius_alpha=self.radius_alpha, norm=norm, y_norm=y_norm, ls_failed=ls_failed, orthogonality=orthogonality, n_iter=new_ys_x)
#              d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

              del H_diag
#TODO: fix this, we just need to write to hist not calculate everything else but we shouldnt check ys for this condition
#TODO: this or the above should be redundant trace and remove redundancy
#          if n_iter >= max_iter or loss == 0:
#            break
          prev_flat_grad = flat_grad.cpu()
          prev_loss = loss
          # normalize the Hessian's direction #TODO: try scaling the Hessian approximation instead of the resultant direction. Can also try to normalize y s and ys in theory inv Hessian computation can overflow (or even underflow) with large history sizes
#TODO: should we be iterating each tensor for norm like in flat_grad?
#          total_norm = torch.abs(d.coalesce().values()).sum().to(self.direction_device) # Move total_norm to direction_device
#          d = dense_to_sparse_flat_tensor(d)
#          loss, _ = self._directional_evaluate(closure, 1., d)
#          print("loss after to sparse: " + str(loss))
#          prev_loss = loss
#          print("DIRECTION: first and last tensors:" + str(d[-10:]) + " " + str(d[:10]))

          ############################################################
          # compute step length
          ############################################################
          # reset initial guess for step size
          # directional derivative
   #TODO: see if we can get bracketing instead to make this faster, e.g. set to 1 so we start t_prev and t at 0,1 this allows for one of the most interesting aspects of L-BFGS: maximum loss reduction with minimal gradient magnitude (CRAM the model information wise) since we would be preferentially bracketing lowest Strong Wolfe points first in terms of step size
          gtd_sparse_product = flat_grad * d.to(self.optimizer_device) # Ensure d is on optimizer_device
          gtd = gtd_sparse_product.sum()  # g * d
          del gtd_sparse_product
          gc.collect()
          torch.cuda.empty_cache()
          d = dense_to_sparse_flat_tensor(d)
          #          prev_flat_grad = prev_flat_grad.to(self.direction_device) # This move is handled before y calculation
          t = self.t
          # directional derivative is below tolerance
          #          if gtd > -tolerance_change:
          #              break

          # optional line search: user function
          ls_func_evals = 0
          if line_search_fn is not None:
              # perform line search, using user function
              if line_search_fn != "strong_wolfe":
                  raise RuntimeError("Only 'strong_wolfe' is supported for line search.")
              else:
                  # No need to clone parameters, _directional_evaluate will handle adding/subtracting
                  def obj_func(t_step, d_direction):
                      return self._directional_evaluate(closure, t_step, d_direction)

                  loss_before_ls = loss
                  flat_grad_before_ls = flat_grad
                  prev_loss = loss

                  success, loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                      obj_func, self.direction_device, t, d, loss, flat_grad, gtd, c2=c2, c1=c1, bracket_shift=bracket_shift, bracket_shove=bracket_shove, capture_min_step=capture_min_step, capture_max_step=capture_max_step, optimizer_device=self.optimizer_device , max_ls = self.max_ls
                  )
                  # TODO: consider the armijo condition here to prevent bonking at higher orders (initial norm of 1).
                  # TODO: fix the needle. Currently this should work since we skip on last iteration anyways but we should be able to take needle on first iter.
              if not success:
                  print("\033[91mLinesearch failure, Rho rewind and skip.\033[0m")
#TODO: this is wrong we arent usinc rho_rewinh
#TODO: consider Rho rewind as a parameter passed into direction approximate that temporarily ignores the top N rho entries, similar to similarity (Since similarity no longer considers the al rho scaled term just q curvature)
# Remove 10 largest rho entries from history if line search fails
                  if len(ro) > 0:
                      magnitudes = torch.stack([
                          (s.values.pow(2).sum() + s.unit_values.pow(2).sum()) * r.item()
                          for s, r in zip(old_stps, ro)
                      ])
                      num_to_remove = min(rho_rewind, len(magnitudes))
                      _, indices_to_remove = torch.topk(magnitudes, num_to_remove, largest=True)
                      
                      for idx in sorted(indices_to_remove.tolist(), reverse=True):
                          if idx < len(old_dirs):
                              old_dirs.pop(idx)
                              old_stps.pop(idx)
                              ro.pop(idx)

#                      print(f"  Removed {num_to_remove} largest and {num_oldest_to_remove} oldest rho entries due to line search failure.")
#TODO: or we could simply set similarity as a momentum like factor for convergence. Possibly even scaling it by the gradient gtd convergence metric with a scalar coefficient hyperparameter.
#TODO: Actually Rho Rewind.. or not?
#                  prev_flat_grad = None # Force gradient search on next iteration
                      ls_failed = True
                      state["ls_failed"] = True # Store ls_failed state
                      # Save the modified history back to state before returning
                      state["old_dirs"] = old_dirs
                      state["old_stps"] = old_stps
                      state["ro"] = ro
                      self.t = 1
                  return orig_loss # Skip this data point
              else: # Strong Wolfe line search succeeded
                  ls_failed = False
                  state["ls_failed"] = False # Store ls_failed state

              if not ls_failed: # If line search (or needle) was successful
                  # Ensure t and d are on the correct device (optimizer_device) for _add_grad
#                  t = t.to(self.optimizer_device)
#                  d = d.to(self.optimizer_device)
                  # _add_grad expects 'update' to be the direction 'd' scaled by 'step_size'
                  # If 'd' is SparseFlatTensor, use _add_sparse_dense with alpha=step_size
                  # If 'd' is dense Tensor, use it directly with alpha=step_size

                  # The _add_grad function itself takes step_size and update.
                  # It applies update to parameters using step_size.
                  # The existing implementation calls p_slice.add(view.view_as(p_slice), alpha=step_size).
                  # If 'update' is a SparseFlatTensor, this would fail.
                  # The prompt implies _add_grad needs the scaling functionality.
                  # For now, we are directly applying the scaled sparse update
                  # in _directional_evaluate. If _add_grad is also meant to
                  # take a SparseFlatTensor as 'update', it would need refactoring.
                  # The current logic in _add_grad handles dense 'update' with alpha.
                  # If 'd' is sparse, _directional_evaluate handles it.
                  # For now, we assume _add_grad might not directly receive sparse 'd'.
                  if isinstance(d, SparseFlatTensor):
                      offset = 0
                      for p in self._params:
                          numel = p.numel()
                          if torch.is_complex(p):
                              p_view = torch.view_as_real(p).view(-1)
                          else:
                              p_view = p.view(-1)
                          # Apply the scaled sparse direction to the dense parameter
                          # using the new function.
                          # --- Key Change: Pass the current offset ---
                          SparseFlatTensor._add_sparse_dense_alpha(d, p_view, alpha=t, offset=offset)
                          offset += numel
                  else: # d is a dense Tensor
                      self._add_grad(t, d)

                  loss_device = self.optimizer_device
                  print(f" \n -----------got stepsize: {t} and loss: \033[92m{loss}\033[0m on device: {loss_device}-----------")
                  # opt_cond = loss <= 0 # This condition is not used later, can be removed if not needed elsewhere
                  self.t = t#.item() # Update self.t with the successful step size

          # update func eval
          current_evals += ls_func_evals
#          state["func_evals"] += ls_func_evals

          ############################################################
          # check conditions
          ############################################################
#          if n_iter == max_iter or loss == 0:
#              break

#          if current_evals >= max_eval:
#              break

          # optimal condition
#          if opt_cond:
#              print("GRAD CONVERGE")
#              break

          # lack of progress
#          if d.mul(t).abs().max() <= tolerance_change:
#              break
#
#          if abs(loss - prev_loss) < tolerance_change:
#              break

#      state["d"] = d
#      state["t"] = t
      state["old_dirs"] = old_dirs
      state["d"] = d
      state["old_stps"] = old_stps
      state["ro"] = ro
#      state["H_diag"] = H_diag
      state["prev_flat_grad"] = prev_flat_grad
      state["ls_failed"] = ls_failed # Store ls_failed state
#      state["prev_loss"] = prev_loss
#      state["n_iter"] = 0 #TODO: MoE equivalent centinuous sparse model using l1 with novel direction per iteration, if we reuse the hessian and there is sparsity the curvature will bias to a lopsided model but is appropriate for l2

      return orig_loss

    def save_history(self, filename):
        """Save FBFGS history to a file."""
        state = self.state[self._params[0]]
        state_dict = self.state[self._params[0]]
        history = {
            "old_dirs": state_dict.get("old_dirs", []),
            "old_stps": state_dict.get("old_stps", []),
            "ro": state_dict.get("ro", []),
            "d": state_dict.get("d", None), # Save direction d
            "prev_flat_grad": state_dict.get("prev_flat_grad", None),
            "flat_grad": state_dict.get("flat_grad", None), # Save flat_grad
            "H_diag": state_dict.get("H_diag", None), # Save H_diag
            "t": self.t, # Save step size t
            "ls_failed": state_dict.get("ls_failed", False), # Save ls_failed state
            "n_iter": state_dict.get("n_iter", 0), # Save iteration count n_iter
        }
        torch.save(history, filename)

    def _move_item_to_device(self, item, device_obj, non_blocking=False):
        if item is None:
            return None
        if isinstance(item, SparseFlatTensor):
            return item.to(device=device_obj, non_blocking=non_blocking)
        else: # torch.Tensor
            return item.to(device=device_obj, dtype=item.dtype, non_blocking=non_blocking)

    def load_history(self, filename):
        """Load FBFGS history from a file."""
        try:
            # Determine map_location for torch.load to prevent initial GPU OOM if history is large
            load_map_location = 'cpu' if self.direction_device == 'cpu' else None
            history = torch.load(filename, map_location=load_map_location)
            state = self.state[self._params[0]]
            device = self.direction_device # Get the device of the model parameters
            # Convert string device to torch.device object for JIT compatibility
            device_obj = torch.device(device)

            # Use list comprehensions for history lists
            state["old_dirs"] = [self._move_item_to_device(item, device_obj, non_blocking=False)
                                 for item in history.get("old_dirs", [])]
            state["old_stps"] = [self._move_item_to_device(item, device_obj, non_blocking=False)
                                 for item in history.get("old_stps", [])]
            state["ro"] = [self._move_item_to_device(item, device_obj, non_blocking=False)
                           for item in history.get("ro", [])]

            # Directly assign and move single tensors
            state["prev_flat_grad"] = self._move_item_to_device(history.get("prev_flat_grad", None), device_obj, non_blocking=False)
            state["flat_grad"] = self._move_item_to_device(history.get("flat_grad", None), device_obj, non_blocking=False)
            state["H_diag"] = self._move_item_to_device(history.get("H_diag", None), device_obj, non_blocking=False)
            state["d"] = self._move_item_to_device(history.get("d", None), device_obj, non_blocking=False)

            t_val = history.get("t", 1) # Load step size t, default to 1 if not found
            if isinstance(t_val, torch.Tensor):
                self.t = t_val.item()
            else:
                self.t = t_val
            state["n_iter"] = history.get("n_iter", 0) # Load iteration count n_iter, default to 0 if not found

            state["ls_failed"] = history.get("ls_failed", False) # Load ls_failed state
            print(f"FBFGS history loaded from {filename}")
        except FileNotFoundError:
            print(f"History file {filename} not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading FBFGS history from {filename}: {e}. Starting from scratch.")
