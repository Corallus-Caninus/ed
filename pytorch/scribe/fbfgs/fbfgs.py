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

from torch.optim.optimizer import Optimizer, ParamsT

#TODO: ensure we are memory efficient. Gather Grads should replace the grads with the view. Im not sure about the implementation but at least we wont allocate a lot of indices for the views? this should not take as much memory as CUDA is saying it does so theres a lot of stuff that can be GC optimized
#TODO: distribution: need to also distributed the norm. Write our own l1 and turn norm hyperparam into a scalar coefficient to ensure the l1 is stable for networks with high parameter count and low type precision.
#TODO: implement SparseFlatTensor addition correctly via AI rendering
#TODO: extract this to a module and begin FBFGS project structuring
#TODO: implement sparse operations where we currently perform dense ops
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


    def __repr__(self):
        return f"SparseFlatTensor(starts={self.starts}, ends={self.ends}, values={self.values}, total_size={self.total_size}, unit_indices={self.unit_indices.numel()})"

    def to_dense(self):
        """
        Converts the sparse tensor representation to a dense PyTorch tensor, including unit indices.
        """
        dense_tensor = torch.zeros(self.total_size, dtype=self.values.dtype, device=self.values.device)

        # Process segments
        if self.starts.numel() > 0: # Check if there are segments to process
            segment_lengths = (self.ends - self.starts).to(torch.long)
            segment_indices_offsets = torch.repeat_interleave(self.starts.to(torch.long), segment_lengths)
            indices = torch.arange(segment_lengths.sum(), device=self.starts.device).to(torch.long)
            segment_lengths_cumsum = segment_lengths.cumsum(0).to(torch.long)
            start_indices = torch.cat([torch.tensor([0], device=self.starts.device), segment_lengths_cumsum[:-1]]).to(torch.long)
            segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
            segment_internal_indices = (indices - start_indices[segment_ids]).to(torch.long)
            segment_indices = (segment_indices_offsets + segment_internal_indices).to(torch.long)
            dense_tensor[segment_indices] = self.values

        # Process unit indices
        if self.unit_indices.numel() > 0: # Check if there are unit indices to process
            dense_tensor[self.unit_indices] = self.unit_values

        return dense_tensor


    def to(self, device: torch.device, non_blocking: bool = False, pin_memory: bool = False):
        """
        Moves all internal tensors to the specified device and returns a new SparseFlatTensor, including unit indices.
        """
        return SparseFlatTensor(
            self.starts.to(device=device, dtype=self.starts.dtype, non_blocking=non_blocking, pin_memory=pin_memory),
            self.ends.to(device=device, dtype=self.ends.dtype, non_blocking=non_blocking, pin_memory=pin_memory),
            self.values.to(device=device, dtype=self.values.dtype, non_blocking=non_blocking, pin_memory=pin_memory),
            self.total_size.to(device=device, dtype=self.total_size.dtype, non_blocking=non_blocking, pin_memory=pin_memory),
            self.unit_indices.to(device=device, dtype=self.unit_indices.dtype, non_blocking=non_blocking, pin_memory=pin_memory),
            self.unit_values.to(device=device, dtype=self.unit_values.dtype, non_blocking=non_blocking, pin_memory=pin_memory)
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

    def rmul(self, scalar):
        """Scalar multiplication."""
        return self.__mul__(scalar)

    @staticmethod
    def add_sparse_dense(sparse_tensor: 'SparseFlatTensor', dense_tensor_arg: Tensor) -> Tensor:
        """
        Adds a SparseFlatTensor to a dense tensor, including unit indices.

        Args:
            sparse_tensor (SparseFlatTensor): The sparse tensor to add.
            dense_tensor (Tensor): The dense tensor to add to.

        Returns:
            Tensor: The dense result of the addition.
        """
        dense_tensor = dense_tensor_arg # Explicitly use dense_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"

        result_dense_tensor = dense_tensor.clone()

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
    def sparse_dot_dense(sparse_tensor_arg: 'SparseFlatTensor', dense_tensor):
        """
        Computes the dot product of a SparseFlatTensor with a dense tensor, optimized for sparsity and unit indices.
        """
        sparse_tensor = sparse_tensor_arg # Explicitly use sparse_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"

        # Initialize dot product with unit indices contribution
        dot_product = torch.tensor(0.0, device=sparse_tensor.values.device, dtype=sparse_tensor.values.dtype)
        if sparse_tensor.unit_indices.numel() > 0:
            unit_values_from_dense = dense_tensor.view(-1)[sparse_tensor.unit_indices]
            dot_product += torch.dot(unit_values_from_dense, sparse_tensor.unit_values)

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
            dot_product += torch.dot(sparse_values_from_dense, sparse_tensor.values)

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


        # 1. Generate segment indices without loops - vectorized approach
        segment_indices_offsets = torch.repeat_interleave(starts_local, segment_lengths)

        # 2. Vectorized value extraction using advanced indexing
        values_local = dense_tensor.view(-1)[segment_indices_offsets]
        total_size_local = torch.tensor(total_size)

    return SparseFlatTensor(starts_local, ends_local, values_local, total_size_local, unit_indices_local, unit_values_local)


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
        xmin_bound_tensor = torch.tensor(xmin_bound)
        xmax_bound_tensor = torch.tensor(xmax_bound)
        return min(max(min_pos_tensor, xmin_bound_tensor), xmax_bound_tensor)
    else:
        return torch.tensor((xmin_bound + xmax_bound) / 2.0)


#TODO: on relaxed wolfe, if loss is reduced from the previous iteration of this data point, accept it (the first iteration is the relaxed wolfe).
#TODO: c3 along with armijo that is c2 but for overconvergence? To prevent early convergence on insta-wolfes? Probably not necessary and would probably slow things down #TODO: cleanup all the AI device mess
def _strong_wolfe(
    obj_func, direction_device, t, d, f, g, gtd, c1=1e-20, c2=0.9, tolerance_change=1e-16, max_ls=5, bracket_shift=(1/3), bracket_shove=(1/3), capture_min_step=1e-4, capture_max_step=100):
#TODO: this irks the mathematician in me.
    if c2 == 0:
      c2 = 0.25
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
    gtd_new = (g_new.to("cuda") * d.to("cuda")).sum() # Keep as scalar tensor
#    g_new = g_new#
#    gtd_new = gtd_new#
    success = False
    is_nan = False

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
#    g_prev = g_prev.to(direction_device)
    done = False
    ls_iter = 0

    t_best = t
    t = torch.tensor(t) # Ensure t is a tensor before the loop
    device = gtd.device
    f_best = torch.tensor(f, device=device)
    g_best = g
#    g = g.to(direction_device)
    gc.collect()
    ls_iter=0
    stall_wolfe=0

#TODO something is broken in insta wolfe. Check the initialization. The same step size doesnt throw when not insta-wolfing
    while ls_iter < max_ls:
#TODO: we can calculate the delta here for insta wolfes and adjust t by the difference, essentially measuring the drift of the interpolation to see if its shifting left or right to try to stay in the min as long as possible over time
#TODO: e.g.: if wolfe is increasing shift up t, if armijo is increasing, shift down t. We may be able to formulate this as a linear equation or a ratio
        # check conditions #TODO: <= for ward condition should be < and just allow first iteration to not check ward condition #TODO: this can increase loss if f_best is greater than current loss (last iteration loss)
        if f_new > (f + c1 * t * gtd)  or (f_new != f_new and is_nan == True): # or f_new >= f_prev: #NOTE: Ward condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
#            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

#TODO: <= for ward condition should be < and just allow first iteration to not check ward condition
        if abs(gtd_new) <= -c2 * gtd and f_new < f_best :
            bracket = [t] # type: ignore[list-item]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            success = True
            print("FAST WOLFE")
            break

        if gtd_new >= 0 :
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
#            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
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
        if f_new != f_new:
          t = torch.tensor(1.)
          min_step = torch.tensor(0.)
          t = _cubic_interpolate(
              t_prev, f_prev, gtd_prev.to("cuda"), t, f_new, gtd_new.to("cuda"), bounds=(min_step, max_step)
          )
          is_nan = True
        t = torch.tensor(t) #.item() # get scalar value from tensor

        # next step
        t_prev = tmp
        f_prev = f_new
#        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        g_prev = g_new.to(direction_device)
        gtd_prev = gtd_new # type: ignore[assignment] # type: ignore[assignment]
        f_new, g_new = obj_func(t, d)
        ls_func_evals += 1 # Increment func evals after new evaluation
        gtd_new = (g_new.to("cuda") * d.to("cuda")).sum() # Keep as scalar tensor
#        g_new = g_new#
        ls_iter += 1
        #RELAXED WOLFE CONDITION
#        cur_c2 =  abs(gtd_new) - -gtd  #TODO: inverted case
#TODO: armijo in relaxed wolfe condition
        if f_new < f_best and done != True and f_new == f_new: #and (f_new <= (f + c1 * t * gtd)) : #  or f_new >= f_prev: #NOTE: Ward condition
#        if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best: # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
#NOTE: prevent using the first iteration, so that we know we fulfilled the armijo condition. Theres a cleaner way to do this
          success = True
          stall_wolfe = 0
          t_best = t
          f_best = torch.tensor(f_new, device=device)
          g_best = g_new.to(direction_device)

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
        if abs(bracket[1] - bracket[0])  < tolerance_change or  stall_wolfe >= 5:   # type: ignore[possibly-undefined]
           print("WOLFE PACK")
           return success, f_best, g_best.to("cuda"), t_best, ls_func_evals
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
        t = torch.tensor(t)
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
                    t = torch.tensor(t - bracket_shove*(t - displacement))
                    print("punt", end = " ")
                else:
                    displacement = min(bracket) + eps
                    t = torch.tensor(t + bracket_shove*(displacement - t))
                    print("punt", end = " ")
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(t, d) # Single evaluation
        ls_func_evals += 1 # Increment func evals
        gtd_prev = gtd_new
        gtd_new = (g_new.to("cuda") * d.to("cuda")).sum() # Keep as scalar tensor

#TODO: something like this. we need to move t by the amount that linearly should bring it into c3 notch if we are over converged.
#TODO: also this should happen before evaluation and probably be an if condition against push/shove routine
#TODO: prompt it..
#        if abs(gtd_new) <= -0.4 * gtd:
#          gtd_delta = gtd_new - gtd_prev
#          t_delta = t - t_prev
#TODO: we also need the distance to the notch.
#          t = t*t_delta/gtd_delta

        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?


        if f_new > (f + c1 * t * gtd)  or f_new >= bracket_f[low_pos] or f_new != f_new: #or f_new > f_best: #NOTE: Ward condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0) # type: ignore[possibly-undefined]
        else:
            if abs(gtd_new) <= -c2 * gtd and f_new < f_best: #NOTE: Ward condition #TODO: Ward condition should be < not <=, it should be based on < and if gtd is under a threshold such that we cant get a gtd delta
                # Wolfe conditions satisfied
                print("STRONG WOLFE")
                success = True
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
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
            if f_new < f_best and f_new == f_new:#and done != True and (f_new <= (f + c1 * t * gtd)) : #  or f_new >= f_prev: #NOTE: Ward condition
#            if (f_new > (f + c1 * t * gtd)) and done != True and f_new < f_best:  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
    #          print("---GOT NEW WOLFE PACK---")
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
        if ls_iter >= max_ls and done != True and success == False: # Return Wolfe pack if max ls reached in zoom phase
          print("WOLFE PACK MAX LS")
          return success, f_best, g_best.to("cuda"), t_best, ls_func_evals


    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return success, f_new, g_new.to("cuda"), t, ls_func_evals


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
        clop: float = 0,
        direction_device: str = 'cpu',
        norm: float = 1.0,
        y_norm: Optional[float] = None
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
            clop=clop,
            direction_device=direction_device,
            norm=norm,
            y_norm=y_norm
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "FBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self.clop = clop
        self.clop = clop
        self.direction_device = direction_device
        self.t = 1

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    # gather flat grads with L1 Normalization and without clopping
#TODO: dont gather, just let ops be distributed
    def _gather_flat_grad(self):
#        if dist.is_initialized():
#            views = []
#            local_grads = []  # List for local flattened gradients
#            for p in self._params:
#                grad_device = p.device
#                torch.nn.utils.clip_grad_value_(p, torch.finfo(p.dtype).max)
#                if p.grad is None:
#                    view = p.new(p.numel()).zero_()
#                elif p.grad.is_sparse:
#                    view = p.grad.view(-1)
#                else:
#                    view = p.grad.view(-1)
#                if torch.is_complex(view):
#                    view = torch.view_as_real(view).view(-1)
#                views.append(view)
#                local_grads.append(view)  # Append to local_grads list
#
#            world_size = dist.get_world_size()
#            gathered_grads = [torch.empty_like(local_grads[0]) for _ in range(world_size)]  # List for gathered gradients
#            dist.all_gather(gathered_grads, local_grads[0])  # Perform all_gather
#
#            grad = torch.cat(gathered_grads, 0)  # Concatenate gathered gradients
#        else:
        views = []
        for p in self._params:
            grad_device = "cuda" #p.device # Get the device of the gradient
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.view(-1) # Move sparse grad to direction_device
            else:
                view = p.grad.view(-1) # Move dense grad to direction_device
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view.to(grad_device))
        grad = torch.cat(views, 0)
#        for p in self._params: # Clip after gathering to ensure all grads are included
#            if p.grad is not None: # Check if p.grad is not None
#                torch.nn.utils.clip_grad_value_(p, torch.finfo(p.dtype).max)
#        finfo = torch.finfo(grad.dtype)
#        grad = torch.nan_to_num(grad, nan=0.0, posinf=finfo.max, neginf=finfo.min)
#        total_norm = torch.linalg.vector_norm(grad, ord=2.).to("cuda") # Move total_norm to direction_device
#####TODO: safenorm for all these. This is most important because of the initial gradient may be vanishing.
#####        total_norm = total_norm + 1e-8
#        grad = grad.div_(total_norm)
        return grad

    # gather flat grads with L1 Normalization and without clopping
#TODO: rename
    def _gather_flat_grad_DEPRECATED(self):
        views = []
        for p in self._params:
            grad_device = p.device # Get the device of the gradient
            torch.nn.utils.clip_grad_value_(p, torch.finfo(p.dtype).max)
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to(self.direction_device).view(-1) # Move sparse grad to direction_device
            else:
                view = p.grad.to(self.direction_device).view(-1) # Move dense grad to direction_device
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        grad = torch.cat(views.to("cuda"), 0)
#        norm_val = torch.linalg.vector_norm(grad, ord=1.)
#        grad = grad/norm_val
#        return torch.cat(views, 0).to(self.direction_device)
        return grad
#TODO: clip out NaN based on dtype max value
#        return grad_raw #.to(self.direction_device)

    # gather flat grads with L2 Normalization
#TODO: rename
    def _gather_norm_flat_grad(self, isClop = True):
        views = []
        total = 0
        for p in self._params:
            torch.nn.utils.clip_grad_value_(p, torch.finfo(p.dtype).max)
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        views = torch.cat(views, 0)
        norm = torch.linalg.vector_norm(views, self.norm)
        views.div_(norm)
#TODO: does l1 need a norm scaling parameter or does it naturally scale since it has to sum to one anyways (values that are essentially 0 dont add anything to the norm so it should automatically balance). We may also want a scaling value since large networks might end up clopping too much or even dropping too much with l1. Can we tune this normal scaling value with the same hyperparameter used for clopping s.t. its a hyperparameter that is proportional to a "sub net size"? Probably cant just be one hyperparameter, but can we pass in values 0>x<1? essetially the l0.5 norm for scaling up a bit to account for precision losses? Test this but likely we need a hyperparameter to scale the norm we got from l1.
#TODO: what if we normaling by the max value and let clopping handle what the l1 would do anyways? we would only need to tune the clopping hyperparameter and would get essentially what we want with l1
        #Clop
#TODO: may be worth taking the top K here to have deterministic memory, do this after clopping to create a floor for allocation since we want to allow very sparse outlier gradients
        if isClop:
          print("gradient elements: " + str((views != 0).sum()) + " total: " + str(views.numel()), end=' ')
          views[torch.logical_and(views > -self.clop,views < self.clop)] = 0
#          views = views.to_sparse()
        return views #.to("cpu")
    #TODO: clip out NaN based on dtype max value
    #        return grad_raw #.to("cpu")

#TODO: we can just clone the bitmask of the sparse gradients since those are the only params we are going to modify
    # def _clone_param(self):
    # #        return [p.clone(memory_format=torch.contiguous_format).to(self.direction_device) for p in self._params]
    #     return [p.clone(memory_format=torch.contiguous_format) for p in self._params] #        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    # def _set_param(self, params_data): #     for p, pdata in zip(self._params, params_data): #         p.copy_(pdata)

    def _add_grad(self, step_size, update, limit_offset: int = -1) -> int:
        offset = 0
        for p in self._params:
#TODO we need to just add logging and fix this so we dont NaN brick the model on evaluation.
#TODO: was >=
#TODO: >= should be correct since we dont apply the NaN slice
            if limit_offset != -1 and offset >= limit_offset:
                break # Stop processing if we've reached or passed the limit_offset

            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()

            current_param_end_offset = offset + numel
            slice_end = min(current_param_end_offset, limit_offset if limit_offset != -1 else current_param_end_offset)
            slice_numel = slice_end - offset

            if slice_numel <= 0:
                offset += numel
                continue

            view = update[offset : offset + slice_numel].to(p.device)
            p_flat = p.view(-1)
            p_slice = p_flat[0:slice_numel]

            p_temp = p_slice.add(view.view_as(p_slice), alpha=step_size)
            if torch.isnan(p_temp).any():
                del p_temp
                torch.cuda.empty_cache()
                return offset # Return the offset where NaN was found
            p_slice.copy_(p_temp) # Only copy if no NaN
            del p_temp
            torch.cuda.empty_cache()
            offset += numel
        return self._numel() # Return total numel if no NaN

    def _directional_evaluate(self, closure, t, d): #TODO: this function is redundant with _directional_evaluate after memory optimization # and is not called anywhere. Removing it.
        # Save current parameters to CPU
        original_params_cpu = [p.detach().clone().cpu() for p in self._params]
        original_params_cpu = [p.pin_memory() for p in original_params_cpu]
        # Apply step: x_new = x_old + t * d
        offset = 0
        for p in self._params:
            numel = p.numel()
            if torch.is_complex(p):
                p_view = torch.view_as_real(p).view(-1)
            else:
                p_view = p.view(-1)
            p_view.add_(d[offset : offset + numel].to(p.device), alpha=t)
            offset += numel

        loss = float(closure())
        flat_grad = self._gather_flat_grad()

        # Restore original parameters from CPU
        for p, original_p_cpu in zip(self._params, original_params_cpu): # type: ignore[possibly-undefined]
            p.copy_(original_p_cpu.to(p.device))

        return loss, flat_grad
    @torch.jit.script
#TODO: what causes direction norm to blow up?
#TODO: if we increase the type precision we may fix the exploding direction which could result in very large and efficient step sizes.
    def sparse_direction_approximate(old_stps: list[SparseFlatTensor], old_dirs: list[SparseFlatTensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, direction_device: str,t: float, clop: float, norm: float, y_norm: float) -> Tensor:
        torch.cuda.synchronize() # Ensure all previous CUDA operations are complete, especially non-blocking transfers to CPU history
        num_old = len(old_dirs)
        hit_miss = str("")
#        similarity = 5e-5
        similarity = 0.

        q = flat_grad.to(torch.float32).to("cuda").neg()
        total_norm = torch.linalg.vector_norm(q, ord=2.).to(torch.float32).to("cuda")
#        total_norm = max(1e-9, total_norm)
        print("q max value: " + str(q.max()))
        if total_norm == float('inf'):
#          total_norm = torch.tensor(1e-9, dtype=torch.float32, device="cuda")
          print("pre-direction l2 norm returned inf")
        q = q.div_(total_norm)

        al = torch.empty(num_old, dtype=q.dtype, device="cuda")
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=direction_device)

        if num_old > 0:
            # Prefetch the first element for the backward loop
            next_sparse_dir_prefetch: SparseFlatTensor = old_dirs[num_old - 1].to(torch.device("cuda"), non_blocking=True)

            for i in range(num_old - 1, -1, -1):
                torch.cuda.synchronize() # Ensure current_sparse_dir is ready
                current_sparse_dir_val = torch.jit.annotate(SparseFlatTensor, next_sparse_dir_prefetch) # Get the prefetched tensor

                if i > 0:
                    next_sparse_dir_prefetch = old_dirs[i - 1].to(torch.device("cuda"), non_blocking=True) # Initiate prefetch for next iteration

                # Create a new SparseFlatTensor with internal values cast to float32
                sparse_dir_i = SparseFlatTensor(
                    current_sparse_dir_val.starts, current_sparse_dir_val.ends, current_sparse_dir_val.values.to(dtype=torch.float32),
                    current_sparse_dir_val.total_size, current_sparse_dir_val.unit_indices, current_sparse_dir_val.unit_values.to(dtype=torch.float32)
                )
                direction_similarity = SparseFlatTensor.sparse_dot_dense(sparse_dir_i, q).item()
                aligned = direction_similarity >= similarity  or direction_similarity <= -similarity
                direction_alignment_mask[i] = aligned
                if direction_alignment_mask[i]:
    #              similarity = similarity + similarity/direction_similarity #TODO: fix this, it should scale based on the difference
    #              similarity = 2*similarity 
                  al[i] = direction_similarity * ro[i].item()
                  sparse_old_dir_scaled = SparseFlatTensor(
                      current_sparse_dir_val.starts, current_sparse_dir_val.ends, current_sparse_dir_val.values.to(dtype=torch.float32),
                      current_sparse_dir_val.total_size, current_sparse_dir_val.unit_indices, current_sparse_dir_val.unit_values.to(dtype=torch.float32)
                  ) * ((-al[i]))
                  q = SparseFlatTensor.add_sparse_dense(sparse_old_dir_scaled, q)
    #              total_norm = torch.linalg.vector_norm(q, ord=2.).to(torch.float32).to("cuda")
    #              q.div_(total_norm)
    #TODO: try l2 here again? The original reasoning was this breaks the math since division of a term isnt distributive
                  hit_miss = hit_miss + str("| ")
                else:
                  hit_miss = hit_miss + str("_ ")

        print("q max value: " + str(q.max()))
#TODO: test this. we are taking a pragmatic appoarch to the observation that direction blows up on convergence but I think we need to slow down convergence e.g.: by taking rho on the l2 instead of orienting rho to the raw gradient/curvature
#TODO: it may be better to crash out on NaN
#        d = torch.nan_to_num(q.mul(H_diag.to(torch.float32)), nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
        d = q.mul(H_diag.to(torch.float32))
#        total_norm = torch.linalg.vector_norm(q, ord=2.).to(torch.float32).to("cuda")
#        q.div_(total_norm)
        del q

        if num_old > 0:
            # Prefetch the first elements for the forward loop
            next_old_dir_prefetch_fwd: SparseFlatTensor = old_dirs[0].to(torch.device("cuda"), non_blocking=True)
            next_old_stp_prefetch_fwd: SparseFlatTensor = old_stps[0].to(torch.device("cuda"), non_blocking=True)

            for i in range(num_old):
                torch.cuda.synchronize() # Ensure current_old_dir and current_old_stp are ready
                current_old_dir_val = torch.jit.annotate(SparseFlatTensor, next_old_dir_prefetch_fwd)
                current_old_stp_val = torch.jit.annotate(SparseFlatTensor, next_old_stp_prefetch_fwd)
                if i < num_old - 1:
                    next_old_dir_prefetch_fwd = old_dirs[i + 1].to("cuda", non_blocking=True)
                    next_old_stp_prefetch_fwd = old_stps[i + 1].to(torch.device("cuda"), non_blocking=True)

                if direction_alignment_mask[i]:
                  old_dir_for_dense = SparseFlatTensor( # Use current_old_dir_val, which is prefetched
                      current_old_dir_val.starts, current_old_dir_val.ends, current_old_dir_val.values.to(dtype=torch.float32), # type: ignore[arg-type]
                      current_old_dir_val.total_size, current_old_dir_val.unit_indices, current_old_dir_val.unit_values.to(dtype=torch.float32)
                  )
                  dot_product_val = SparseFlatTensor.sparse_dot_dense(old_dir_for_dense, d)
                  alpha_val = al[i] - dot_product_val * ro[i].item()
                  sparse_old_stp_scaled = SparseFlatTensor( # Use current_old_stp_val, which is prefetched
                      current_old_stp_val.starts, current_old_stp_val.ends, current_old_stp_val.values.to(dtype=torch.float32),
                      current_old_stp_val.total_size, current_old_stp_val.unit_indices, current_old_stp_val.unit_values.to(dtype=torch.float32)
                  ) * (alpha_val)
                  d = SparseFlatTensor.add_sparse_dense(sparse_old_stp_scaled, d)
#TODO: try removing normalization from only second loop
#              total_norm = torch.linalg.vector_norm(d, ord=2.).to(torch.float32).to("cuda")
#              d.div_(total_norm)

#        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        print(hit_miss)
        total_norm = torch.linalg.vector_norm(d, ord=norm).to(torch.float32).to("cuda")
#        total_norm = max(1e-9, total_norm)
#        if total_norm == float('inf'):
#            total_norm = torch.linalg.vector_norm(d, ord=float("inf")).to(torch.float32).to("cuda")
#            d = d.div_(total_norm)
#            total_norm = torch.linalg.vector_norm(d, ord=norm).to(torch.float32).to("cuda")
        print("max value pre-norm direction: " + str(d.max()))
        d = d.div_(total_norm)

        d = d.to(torch.float16)
        mask = torch.logical_and(d > -clop, d < clop)
        d[mask] = 0
#        d = d.mul_(total_norm)
#        print("direction elements: " + str((d != 0).sum()) )
        print("total_norm: " + str(total_norm))
        del mask
        return d

    @torch.jit.script
    def dense_direction_approximate(old_stps: list[Tensor], old_dirs: list[Tensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, direction_device: str,t: float, clop: float, norm: float) -> Tensor:
        torch.cuda.synchronize() # Ensure all previous CUDA operations are complete, especially non-blocking transfers to CPU history
        num_old = len(old_dirs)
        hit_miss = str("")
        similarity = 0.
#TODO: underflow also this should be better formulated and we should try to avoid another hyperparam but arbitrary literals are worse than hyperparams
        if t < 1:
          similarity = similarity/t
        q = flat_grad.neg().to("cuda")
        total_norm = torch.linalg.vector_norm(q, ord=2.).to("cuda") # Move total_norm to direction_device
        q = q.div_(total_norm)
#        mask = torch.logical_and(q > -clop, q < clop) #TODO: extract to sub_variance hyperparameter

        al = torch.empty(num_old, dtype=q.dtype, device="cuda") # Initialize al as tensor
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device="cuda")

        if num_old > 0:
            # Prefetch the first element for the backward loop
            next_old_dir_prefetch_bwd: Tensor = old_dirs[num_old - 1].to("cuda", non_blocking=True)

            for i in range(num_old - 1, -1, -1):
                torch.cuda.synchronize() # Ensure current_old_dir is ready
                current_old_dir_val = torch.jit.annotate(Tensor, next_old_dir_prefetch_bwd)
                if i > 0:
                    next_old_dir_prefetch_bwd = old_dirs[i - 1].to("cuda", non_blocking=True)
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

        d = torch.nan_to_num(q.mul(H_diag), nan=0.0, posinf=0.0, neginf=0.0)
        be_i = torch.empty_like(d, dtype=q.dtype, device="cuda") # Preallocate be_i for second loop
        del q

        if num_old > 0:
            # Prefetch the first elements for the forward loop
            next_old_dir_prefetch_fwd: Tensor = old_dirs[0].to("cuda", non_blocking=True)
            next_old_stp_prefetch_fwd: Tensor = old_stps[0].to("cuda", non_blocking=True)

            #TODO: vectorize alignment mask here since its immutable
            for i in range(num_old):
                torch.cuda.synchronize() # Ensure current_old_dir and current_old_stp are ready
                current_old_dir_val = torch.jit.annotate(Tensor, next_old_dir_prefetch_fwd)
                current_old_stp_val = torch.jit.annotate(Tensor, next_old_stp_prefetch_fwd)
                if i < num_old - 1:
                    next_old_dir_prefetch_fwd = old_dirs[i + 1].to("cuda", non_blocking=True)
                    next_old_stp_prefetch_fwd = old_stps[i + 1].to("cuda", non_blocking=True)

                if direction_alignment_mask[i]: # Check alignment for current index
                  be_i.copy_((current_old_dir_val * d)) # Use current_old_dir_val
                  alpha_val = al[i] - be_i.sum() * ro[i].item() # Use al[i] and ro[i]
                  d = d + (current_old_stp_val * (alpha_val)) # Use current_old_stp_val

        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        print(hit_miss)
#TODO: we may increase efficacy and reduce tearing by supplemnting clopping with a lower order norm
        total_norm = torch.linalg.vector_norm(d, ord=norm).to("cuda")
        d = d.div_(total_norm)
        direction = d
#        mask = torch.logical_and(direction > -clop, direction < clop) #TODO: extract to sub_variance hyperparameter
#        direction[mask] = 0
#        d = direction.to_sparse()
#        print("direction elements: " + str((direction != 0).sum()) )
#        del mask # DEL 9: mask is no longer needed
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

      # NOTE: FBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      state = self.state[self._params[0]]

      # evaluate initial f(x) and df/dx
      orig_loss = closure()
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
#      flat_grad = None
#      prev_flat_grad = None
#
      n_iter = 0
#      d = flat_grad.neg() # Initialize d on direction_device
      first_param = next(self.param_groups[0]['params'].__iter__())
      t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)
      ls_failed = False
      # optimize for a max of max_iter iterations
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
          if  n_iter == 1 or prev_flat_grad is None:
#          if prev_flat_grad is None:
              restart = False
#TODO: use the proper flat_grad (the l1 instead of l2) here since we dont calculate direction first
              print("RESET (n_iter=1 or prev_flat_grad is None)")
              flat_grad = self._gather_flat_grad()
#TODO: clip_grad_norm by the l1 norm for a max norm of 1e9
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9)
              if flat_grad.abs().max() <= tolerance_grad: #TODO: check if this is even possible given normalization.
                return orig_loss
              H_diag = 1
              H_diag = torch.tensor(H_diag)
#TODO: t shouldnt be 1 here for insta-wolfes
#              t = 1
#              self.t = 1.
#              if len(old_dirs) > 0 and prev_flat_grad is not None:
#                if self.clop == 0:
#                  d = self.dense_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, clop=self.clop, norm=norm)
#                else:
#                  d = self.sparse_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, clop=self.clop, norm=norm, y_norm = y_norm)
#              else:
              d = self._gather_flat_grad().neg()
              #TODO: should we also do norm float("inf") here to match direction S?
              total_norm = torch.linalg.vector_norm(d, ord=norm) # Move total_norm to direction_device
#              total_norm = max(1e-9, total_norm)
              #Handle type precision overflow for L1-likes
              if total_norm == float('inf'):
                total_norm = torch.linalg.vector_norm(d, ord=float("inf")) # Move total_norm to direction_device
                d = d/total_norm
                total_norm = torch.linalg.vector_norm(d, ord=norm) # Move total_norm to direction_device
              print("d norm: " + str((total_norm)) )
              d = d/total_norm
              d[torch.logical_and(d > -self.clop,d < self.clop)] = 0
		#NOTE: end of else

#              d = d.to_sparse()
              d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
#              d = d*total_norm
              gc.collect()
#              print("d elements: " + str((d.values() != 0).sum()) )
              print("direction elements: " + str((d != 0).sum()) )
          else:
#              total_norm_grad = torch.linalg.vector_norm(flat_grad, ord=2.) # Move total_norm to direction_device
#              norm_flat_grad = flat_grad/total_norm_grad
#
#              total_norm_prev_grad = torch.linalg.vector_norm(prev_flat_grad, ord=2.) # Move total_norm to direction_device
#              prev_norm_flat_grad = prev_flat_grad/total_norm_prev_grad # Creates new tensor

              # Calculate y_dense using clone and in-place operations to reduce allocations
#TODO: clip flat_grad and prev_flat_grad here respectively.
              # Apply L2 norm clipping to flat_grad and prev_flat_grad
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9)
#              if prev_flat_grad is not None:
#                  torch.nn.utils.clip_grad_norm_(prev_flat_grad, max_norm=1e9)
#TODO: clip flat_grad and prev_flat_grad here respectively.
              # prev_flat_grad is moved to CUDA for calculation, then immediately deleted
              # to free up memory.
              y_dense = flat_grad.clone() # Allocate y_dense once by cloning norm_norm_flat_grad
              y_dense.sub_(prev_flat_grad.to("cuda")) # Perform subtraction in-place (avoids new tensor for subtraction result)
              s_dense = (d.mul(t)) # Define s_dense here
              ys = y_dense.dot(s_dense)

              original_y_dtype = y_dense.dtype
              y_dense_float32 = y_dense.to(torch.float32)
              norm_y_dense = torch.linalg.vector_norm(y_dense_float32, ord=2.)
#              norm_y_dense = torch.linalg.vector_norm(y_dense_float32, ord=2.)
#TODO: it may be of note that doing selection on the raw y may remove some of the late convergence aspects of the l2 distribution despite being a sample of the l2 distribution. We may need to normalize first (but keep rho on raw) for the y selection
#              norm_y_dense = max(1e-9, norm_y_dense)
              y_dense_float32.div_(norm_y_dense) 
              norm_s = torch.linalg.vector_norm(s_dense, ord=2.)
#TODO: try without norming d now that we have decent alpha deflection
#              ys = y_dense_float32.dot(s_dense.div(norm_s).to(torch.float32))  # Calculate ys here after s is SparseFlatTensor
#              ys = y_dense_float32.dot(s_dense.to(torch.float32))  # Calculate ys here after s is SparseFlatTensor
#              ys = 100*ys #I hate everything about this.. at least make it max(1, 100-len(old_dirs))..
#TODO: would ys * norm_y_dense make sense? since this is essentially a metric of how lossy the curvature is? possibly with a hyperparameter scalar coefficient?
#              ys = 100*ys #I hate everything about this.. at least make it max(1, 100-len(old_dirs))..
              torch.cuda.empty_cache()

              s_mask = (s_dense != 0)
              ys_dense = y_dense.clone()
              ys_dense[~s_mask] = 0

              #Shotgun noise
              norm_y = torch.linalg.vector_norm(y_dense_float32, ord=y_norm)
              y_dense_float32.div_(norm_y)
              y_dense.copy_(y_dense_float32.to(original_y_dtype))
              del y_dense_float32
              # Apply clopping to y_dense
              if self.clop != 0:
                  y_dense_mask = torch.logical_and(y_dense > -self.clop, y_dense < self.clop)
                  y_dense[y_dense_mask] = 0
                  del y_dense_mask
              y_dense.mul_(norm_y)

#TODO: this isnt working, can verify by setting norm higher than y norm, y norm wont be at least d numel
#TODO: fix this.
#              s_mask = (s_dense != 0)
#              ys_dense = y_dense.clone()
#              ys_dense[~s_mask] = 0
#              norm_y = norm if y_norm is None else y_norm
##              print("y_norm elements: " + str((y_dense != 0).sum()))
              y_mask = (y_dense == 0)
              ys_mask = torch.logical_and(s_mask, y_mask)
              ys_dense[~ys_mask] = 0
              print("y dense pre s-mask " + str((y_dense != 0).sum()))
              print("s mask: " + str((s_mask!=0).sum()))
              y_dense.add_(ys_dense)
              print("y dense + s_mask  " + str((y_dense != 0).sum()))
#              TODO: should we instead take the l2 of s(t) to keep everything in the same norm order in the approximation?
#              s_dense = d #TODO: again? with letting I first loop scale


#              ys = y_dense.mul(norm_y_dense).dot(s_dense.div(norm_s))
#              ys = y_dense.mul(norm_y_dense).dot(s_dense)
              norm_yf = torch.linalg.vector_norm(y_dense, ord=2.)
              y_dense.div_(norm_yf) 
#              ys = y_dense.dot(s_dense.div(norm_s))
#              ys = y_dense.dot(s_dense)

#              s_dense.div_(norm_s)
#              yf = self._numel() / (y_dense != 0).sum()
#              y_dense.div_(yf)

#              norm_y_dense = norm_y_dense * yf #if the full vector is the unit distance, this should be proportional
              del ys_dense
              del ys_mask
              del y_mask
              torch.cuda.empty_cache()
              gc.collect()

#TODO consider scale aware thresholding s.t. ys >= 1e-1* s.dot(s).sqr()
#TODO: maybe this should be rho so we dont throw off the Hessian diag
#              yf = self._numel() / (y_dense != 0).sum()
#              ys = ys * yf
#              ys = ys * 1e2
#NOTE: 0.1 is approx 368/30Mill. we may have a better way to formulate this by clipping according to an inverse of yf
#              yf =  (y_dense != 0).sum() / self._numel()
#              if ys <= 0.1 and ys > 0 and t > 1: #NOTE: was 0.1
#                ys = 0.1

#TODO: ys = Y*S was here
              print(f"ys: {ys}")
#              s_dense = s_dense/total_norm_s
#              s_dense[torch.logical_and(s_dense > -self.clop,s_dense < self.clop)] = 0

#              if self.clop != 0:
              y = dense_to_sparse_flat_tensor(y_dense)
              s = dense_to_sparse_flat_tensor(s_dense)
#              else:
#                y = y_dense
#                s = s_dense
              print("d-delta elements: " + str((d.to_dense() != 0).sum()) + " total: " + str(d.to_dense().numel()), end=' ')
              print("S elements: " + str((s_dense != 0).sum()) + " total: " + str(s_dense.numel()), end=' ')
              print("y-delta elements: " + str((y.to_dense() != 0).sum()) + " total: " + str(y.to_dense().numel()), end=' ')
#TODO: this is correct, but maybe there is something more elegant. Possibly reduce based on the mean or the l1/l2 distribution with a hyperparameter. This can be modeled as a outlier distribution problem. We want to maximize Rho so only remove what we need to stabilize the direction-- how we quantify this is TODO
#TODO: this is arguably better than similarity. I wonder if recency matters, such as remove the oldest entries of large Rho (but more elegant)
#TODO: maybe we can even have a weighted pop for the sliding window that considers both the recency and magnitude of the Rho entries? This is all observations on something that needs to be fundamentally quantified.
              if  ys >= 1e-3  and t >=1:
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
                    while cpu_ram_available <= history_size: # If available RAM is less than history_size
                        cpu_ram_available = psutil.virtual_memory().available / (1024**3)
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                        gc.collect()
                  except Exception as e:
                    print(f"CPU RAM check failed: {e}. Falling back to default memory management.")
                print(f"L-BFGS history popped. History size reduced to: {len(old_dirs)}")
                torch.cuda.empty_cache() # Clear cache before history update
                # store new direction/step
                old_dirs.append(y.to(self.direction_device, non_blocking=True, pin_memory=True)) # Store y as SparseFlatTensor
                old_stps.append(s.to(self.direction_device, non_blocking=True, pin_memory=True)) # Store s as SparseFlatTensor
                ro.append(torch.tensor([(1. / ys)], device=self.direction_device, non_blocking=True, pin_memory=True)) # NOTE: was cpu #TODO: can we include information on convergence here. This may be an observation of the approximation accuracy. Also consider the alignment (gtd being as close to zero as possible). essentially we would be scaling how much the approximation is influenced by an entry based on its ability to converge.
                state["old_stps"] = old_stps
                state["ro"] = ro
                state["old_dirs"] = old_dirs
#TODO: break here on n_iters
              if n_iter > max_iter or loss == 0:
                break
              if flat_grad.abs().max() <= tolerance_grad: #TODO: check if this is even possible given normalization. 
                return orig_loss
              # Update scale of initial Hessian approximation
# TODO: was this also shifted? check the original implementation
              if ys > 0:
                y_squared = y_dense.dot(y_dense)
                H_diag = ys / y_squared # (y*y)
                del y_squared
              else:
                H_diag = 1.
                H_diag = torch.tensor(H_diag)
#              H_diag = ys #TODO: just 1?
#              H_diag = ys #TODO: just 1?
              gc.collect()

              y = y #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              ys = ys #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.

              del y
              del s
              gc.collect()

              # compute the approximate (L-BFGS) inverse Hessian
              # multiplied by the gradient
              num_old = len(old_dirs)
              gc.collect()
              torch.cuda.empty_cache()

              flat_grad = self._gather_flat_grad()
              gc.collect()
              torch.cuda.empty_cache()
#TODO: may need to try this again? the hessian doesnt pertain as much given that the next direction is likely orthogonal to the last.
#TODO: it might make sense to divide by the history size so we keep curvature normalized to prevent explosions in direction approx.
#              H_diag = 1
#              H_diag = torch.tensor(H_diag)
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9)
#              if self.clop == 0:
#                d = self.dense_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, clop=self.clop, norm=norm)
#              else:
              d = self.sparse_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, clop=self.clop, norm=norm, y_norm=y_norm)
              gc.collect()
              d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
              torch.cuda.empty_cache()

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
#          d = d.to_sparse() # Convert to sparse here, before topk
#          print("DIRECTION: first and last tensors:" + str(d[-10:]) + " " + str(d[:10]))

          ############################################################
          # compute step length
          ############################################################
          # reset initial guess for step size
          # directional derivative
   #TODO: see if we can get bracketing instead to make this faster, e.g. set to 1 so we start t_prev and t at 0,1 this allows for one of the most interesting aspects of L-BFGS: maximum loss reduction with minimal gradient magnitude (CRAM the model information wise) since we would be preferentially bracketing lowest Strong Wolfe points first in terms of step size
#          flat_grad = self._gather_norm_flat_grad(1, True) TODO: is this right?
          gtd_sparse_product = flat_grad.to("cuda") * d.to("cuda")
          gtd = gtd_sparse_product.sum()  # g * d
          del gtd_sparse_product
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

                  gc.collect()
                  prev_loss = loss

                  success, loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                      obj_func, self.direction_device, t, d, loss, flat_grad, gtd, c2=c2, c1=c1, bracket_shift=bracket_shift, bracket_shove=bracket_shove, capture_min_step=capture_min_step, capture_max_step=capture_max_step
                  )
                  # TODO: consider the armijo condition here to prevent bonking at higher orders (initial norm of 1).
                  # TODO: fix the needle. Currently this should work since we skip on last iteration anyways but we should be able to take needle on first iter.
              Needle = False
              if not success:  # TODO: we chase misprinted lines
                  # Line search failed. Remove the largest rho entry from history.
                  if len(ro) > 0:
                      removed_count = 0
                      # Iterate backward to safely remove elements without affecting indices of unprocessed elements
                      for i in range(len(ro) - 1, -1, -1):
                          if ro[i].item() > 10.0: # 1 / 1e-1 = 10.0
                              old_dirs.pop(i)
                              old_stps.pop(i)
                              ro.pop(i)
                              removed_count += 1
                      if removed_count > 0:
                          print(f"Removed {removed_count} rho entries > 10.0 from history. New history size: {len(ro)}")
                      else:
                          print("No rho entries > 10.0 found to remove.")
#TODO: remove the largest rho entry from the history (s, y and rho)
                  if ls_failed:  # TODO: we chase misprinted lines
                      return orig_loss # Skip data point if line search failed and needle subroutine would be triggered
                      t = 1  # Reset t to 1 for after needling
                      # TODO: instead of topk, iteratively reduce the norm order and check if loss is reducing or equivalent then increase step size until loss doesnt reduce and repeat
                      # TODO: if we cant decrease at all, we skip the data point, currently loss can increase here.
                      best_overall_needle_loss = prev_loss  # Initialize best_overall_needle_loss (loss before needle)
                      print("saddle-search subroutine..")
                      Needle = True
                      # Capture the negative gradient once before the outer loop # Use the flat_grad from before line search (captured before the main line search attempt)
                      initial_neg_grad = flat_grad.neg().clone()

                      best_overall_d_needle = None  # Store the direction that achieved the best overall loss
                      best_overall_t = torch.tensor(0.0, dtype=first_param.dtype, device=first_param.device)  # Store the step size that achieved the best overall loss

                      needle_norm_order = 1.  # Start with L1 norm - 0.3

                      needle_loss_reduced = False  # Flag to track if needle reduced loss
                      # Outer loop: Decrease norm order until overall loss is reduced or underflow
                      while not needle_loss_reduced and needle_norm_order >= 0:  # Continue until overall reduction or norm order invalid
                          # Start with the initial negative gradient and normalize it
                          d_needle = initial_neg_grad.clone()
                          print(f"  Needle norm order: {needle_norm_order:.2f}", end=' ')
                          print(f"  Needle norm order: {needle_norm_order:.2f}", end=' ')
                          current_norm = torch.linalg.vector_norm(d_needle, ord=needle_norm_order)

                          if current_norm < 1e-9 or needle_norm_order < 0:  # Break outer loop if norm too small or order negative
                              print("Needle norm too small or order negative, breaking outer loop.")
                              break
                          d_needle.div_(current_norm)

                          # --- Inner Loop Starts Here ---
                          # Evaluate loss and gradient at step 1.0 for this norm order
                          current_step_t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)  # Start step size for this norm order iteration
                          # current_step_t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)  # Start step size for inner loop
                          # current_step_t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)  # Start step size for inner loop
                          loss_at_step_1, grad_at_step_1 = self._directional_evaluate(closure, current_step_t, d_needle)
                          gtd_at_step_1 = (grad_at_step_1.to("cuda") * d_needle.to("cuda")).sum()
                          loss_baseline_for_step_increase = loss_at_step_1  # Baseline for Armijo and loss reduction check

                          print(f"  Step size 1.0 with norm order {needle_norm_order:.2f}, Loss: {loss_at_step_1}, GTD: {gtd_at_step_1}")

                          # Check if step 1.0 is a descent direction and improved the overall best loss found so far
                          if gtd_at_step_1 < 0 and loss_at_step_1 <= best_overall_needle_loss:
                              print(f"  Loss reduced at step 1.0 for norm order {needle_norm_order:.2f}. Exploring larger steps.")
                              # Update overall best with step 1.0 result
                              best_overall_needle_loss = loss_at_step_1
                              best_overall_d_needle = d_needle.clone()  # Store the normalized direction
                              best_overall_t = current_step_t.clone()  # Store the step size (1.0)
                              needle_loss_reduced = True  # Mark that we've found at least one reduction

                              # Now, try increasing step size starting from 2.0
                              current_step_t = torch.tensor(2.0, dtype=first_param.dtype, device=first_param.device)

                              while True:  # Inner loop: Iteratively increase step size
                                  # Add a safeguard against unbounded step size before applying
                                  if current_step_t > 1e10:  # Arbitrary large number, could be a hyperparameter
                                      print(f"    Step size {current_step_t:.4f} exceeded max limit, stopping step increase.")
                                      break  # Break inner loop

                                  # Apply step
                                  # We need to evaluate at x_init_needle + current_step_t * d_needle
                                  # _directional_evaluate handles adding/removing the step and evaluating closure
                                  # It also returns the gradient at the new point, which we don't currently use here, but it's part of the function signature.
                                  # It also returns the gradient at the new point, which we don't currently use here, but it's part of the function signature.
                                  current_loss_at_step, _ = self._directional_evaluate(closure, current_step_t, d_needle)
                                  # Evaluate loss at the new point # Evaluate loss # Evaluate loss # Evaluate loss # Evaluate loss
                                  # Evaluate loss
                                  # Undo step
                                  print(f"    Trying step size {current_step_t:.4f} with norm order {needle_norm_order:.2f}, Loss: {current_loss_at_step}")

                                  # Check if this step improved the overall best loss
                                  if current_loss_at_step < best_overall_needle_loss:
                                      best_overall_needle_loss = current_loss_at_step
                                      best_overall_d_needle = d_needle.clone()  # Store the normalized direction
                                      best_overall_t = current_step_t.clone()  # Store the step size
                                      needle_loss_reduced = True  # Set overall success flag
                                      # No need to update needle_loss_reduced here, it's already True

                                  # Check the continuation condition: Armijo holds
                                  armijo_holds = current_loss_at_step <= loss_baseline_for_step_increase + c1 * current_step_t * gtd_at_step_1

                                  if armijo_holds:
                                      # Armijo holds, try larger step
                                      current_step_t *= 2  # Increase step size (e.g., double)
                                  else:
                                      # Armijo failed, stop increasing step size for this norm order
                                      print(f"    Armijo failed for norm order {needle_norm_order:.2f}, stopping step increase.")
                                      break  # Break inner loop
                                  # --- Inner Loop Ends Here ---
                          elif gtd_at_step_1 > 0:
                              # Step 1.0 is not a descent direction for this norm order.
                              print(f"  Step size 1.0 is not a descent direction (GTD >= 0) for norm order {needle_norm_order:.2f}. Skipping step increase.")
                              # No inner loop for step increase if not a descent direction.
                          else:
                              # Step 1.0 is a descent direction (GTD < 0) but did not reduce overall loss.
                              print(f"  Step size 1.0 is a descent direction (GTD < 0) but increased overall loss for norm order {needle_norm_order:.2f}. Skipping step increase.")
                              # No inner loop for step increase if step 1.0 didn't reduce overall loss.

                          # After inner loop (or if skipped), reduce norm order for the next outer iteration
                          needle_norm_order -= 0.3  # type: ignore[operator]

                      if needle_loss_reduced:
                          # Apply the best step found only if loss was reduced
                          self._add_grad(best_overall_t, best_overall_d_needle)  # Use the best step size and best direction
                          loss = best_overall_needle_loss  # Update the main loss
                          print(f" \n -----------Applied needle step with size: {best_overall_t:.4f} and final loss: \033[92m{loss}\033[0m-----------")
                          ls_failed = False  # Needle succeeded in reducing loss
                      else:
                          # Needle failed to reduce loss, skip the step
                          print(f" \n -----------Needle subroutine failed to reduce loss. Skipping step.-----------")
                          # Parameters remain at x_init_needle (which is the state before needle)
                          ls_failed = True  # Indicate that no successful step was found # This line is redundant as we return
                          return orig_loss
#                      del prev_flat_grad
                      del initial_neg_grad
                      if best_overall_d_needle is not None: del best_overall_d_needle
                      if best_overall_t is not None: del best_overall_t
                      del d_needle  # d_needle is cloned inside the loop, but the last one might persist
                      # del x_init_needle # x_init_needle is no longer used
                      torch.cuda.empty_cache()
                      gc.collect()

                  print("\033[91mLinesearch failure, resetting..\033[0m")
                  # If needle search also failed to reduce loss, reset history
                  ls_failed = True
              else: # Line search succeeded
                  ls_failed = False

          # TODO: I dont like having to do this but we want l2 for the direction selection.
          # TODO: dont reset the Hessian if we are using prev step size since one iteration may be insufficient to bracket down
          #                if "old_dirs" in state:
          #                  state["old_dirs"].clear()
          #                  state["old_stps"].clear()
          #                  state["ro"].clear()
          # TODO: dont clear these? may leak here
          #                old_dirs = []
          #                old_stps = []
          #                ro = []
          #                state["n_iter"] = 0
          #              flat_grad = flat_grad.to("cuda")
              if ls_failed and Needle == False:  # and Needle == False:
                  flat_grad = prev_flat_grad
                  prev_flat_grad = None
              else:
                  self.t = t
          if not ls_failed:
              first_param = next(self.param_groups[0]['params'].__iter__())
              t = t.to(first_param.device)
              d = d.to(first_param.device)
              self._add_grad(t, d)
              loss_device = d.device
              # TODO: fix this print (its wrong)
              print(f" \n -----------got stepsize: {t} and loss: \033[92m{loss}\033[0m on device: {loss_device}-----------")  # Use best_needle_loss
              opt_cond = loss <= 0  # TODO: this should be one order of magnitude above the minimum since we start getting convergence problems when we are very close to the min of precision # Use best_needle_loss

          #              opt_cond = flat_grad.abs().max() <= tolerance_grad #TODO: check if this is even possible given normalization. Once verified, rename to point break
          #              opt_cond = opt_cond or loss <= 0 #TODO: this should be one order of magnitude above the minimum since we start getting convergence problems when we are very close to the min of precision
          #         else:
          #              # no line search, simply move with fixed-step
          #              first_param = next(self.param_groups[0]['params'].__iter__())
          ##              t = t.to(first_param.device)
          #              d = d.to(first_param.device)
          #              self._add_grad(t, d)
          #              if n_iter != max_iter:
          #                  # re-evaluate function only if not in last iteration
          #                  # the reason we do this: in a stochastic setting,
          #                  # no use to re-evaluate that function here
          #                  with torch.enable_grad():
          #                      loss = float(closure())
          #                  flat_grad = self._gather_flat_grad()
          #                  opt_cond = flat_grad.abs().max() <= tolerance_grad
          #                  ls_func_evals = 1

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
#TODO: we may not need this, just let it hit epsilon grad or zero grad for number of iteration times?
#TODO: also, dont exit on loss < 1e-5 as above, let that point break (loss <= 0) condition
#          if opt_cond:
#              print("GRAD CONVERGE")
#              break

          # lack of progress
#          if d.mul(t).abs().max() <= tolerance_change:
#              break
#
##TODO: this contition may be not appropriate given relaxed wolfe condition.
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
            "n_iter": state_dict.get("n_iter", 0), # Save iteration count n_iter
        }
        torch.save(history, filename)

    def load_history(self, filename):
        """Load FBFGS history from a file."""
        try:
            history = torch.load(filename)
            state = self.state[self._params[0]]
            device = self.direction_device # Get the device of the model parameters
            # Convert string device to torch.device object for JIT compatibility
            device_obj = torch.device(device)
            pin_mem_flag = (device_obj.type == 'cpu')

            state["old_dirs"] = [tensor.to(device_obj, pin_memory=pin_mem_flag) for tensor in history.get("old_dirs", [])] # Load history and move to direction_device
            state["old_stps"] = [tensor.to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) for tensor in history.get("old_stps", [])] # Load history and move to direction_device
            state["ro"] = [tensor.to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) for tensor in history.get("ro", [])] # Load history and move to direction_device
            state["prev_flat_grad"] = history.get("prev_flat_grad", None) # Load history
            state["flat_grad"] = history.get("flat_grad", None) # Load flat_grad
            state["H_diag"] = history.get("H_diag", None) # Load H_diag #TODO: this should be direction_device
            state["d"] = history.get("d", None) # Load direction d
            t_val = history.get("t", 1) # Load step size t, default to 1 if not found
            if isinstance(t_val, torch.Tensor):
                self.t = t_val.item()
            else:
                self.t = t_val
            state["n_iter"] = history.get("n_iter", 0) # Load iteration count n_iter, default to 0 if not found

            # Move other state tensors to the direction_device with non_blocking and pin_memory
            if state["prev_flat_grad"] is not None:
                state["prev_flat_grad"] = state["prev_flat_grad"].to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) # Move prev_flat_grad to direction_device if it exists
            if state["d"] is not None:
                state["d"] = state["d"].to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) # Move d to direction_device if it exists
            if state["flat_grad"] is not None:
                state["flat_grad"] = state["flat_grad"].to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) # Move flat_grad to direction_device if it exists
            if state["H_diag"] is not None:
                state["H_diag"] = state["H_diag"].to(device_obj, non_blocking=True, pin_memory=pin_mem_flag) # Move H_diag to direction_device if it exists
            print(f"FBFGS history loaded from {filename}")
        except FileNotFoundError:
            print(f"History file {filename} not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading FBFGS history from {filename}: {e}. Starting from scratch.")

    def _compute_direction(self, n_iter, flat_grad, prev_flat_grad, d, t, old_dirs, old_stps, ro, loss, state):
        """Compute the L-BFGS search direction."""
        if n_iter == 1:
            print("RESET")
            d = flat_grad.neg().to("cuda") # Move to calculation device
            H_diag = torch.tensor(1.0).to("cuda") # Move to calculation device
            t = 1
            gc.collect()
        else:
            torch.cuda.empty_cache()
            flat_grad = flat_grad.to("cuda") # Move flat_grad to calculation device
            if prev_flat_grad is not None:
                prev_flat_grad = prev_flat_grad.to("cuda") # Move prev_flat_grad to calculation device
            y = flat_grad.sub(prev_flat_grad)
            s = (d.mul(t)).to("cuda") # Move s to calculation device
            ys = (y * s).sum()

            if ys > 1e-32:
                if torch.device("cuda").type == 'cuda' and torch.cuda.is_available(): # Use torch.device("cuda").type
                    try:
                        cuda_memory_allocated = torch.cuda.memory_allocated(device=torch.device('cuda')) / 1000000000
                        print(f"CUDA memory allocated: {cuda_memory_allocated} GB, history_size: {self.history_size} GB")
                        while cuda_memory_allocated >= self.history_size:
                            cuda_memory_allocated = torch.cuda.memory_allocated(device=torch.device('cuda')) / 1000000000
                            old_dirs.pop(0)
                            old_stps.pop(0)
                            ro.pop(0)
                    except Exception as e:
                        print(f"CUDA memory check failed: {e}. Falling back to psutil.")
                elif torch.device("cuda").type == 'cpu': # Use torch.device("cuda").type
                    try:
                        cpu_ram_available = psutil.virtual_memory().available / (1024**3)
                        print(f"CPU RAM available: {cpu_ram_available} GB, history_size: {self.history_size} GB")
                        while cpu_ram_available <= self.history_size:
                            cpu_ram_available = psutil.virtual_memory().available / (1024**3)
                            old_dirs.pop(0)
                            old_stps.pop(0)
                            ro.pop(0)
                    except Exception as e:
                        print(f"CPU RAM check failed: {e}. Falling back to default memory management.")
                print(f"L-BFGS history popped. History size reduced to: {len(old_dirs)}")
                torch.cuda.empty_cache()

                if self.clop != 0:
                  y_sparse = y.to_sparse()
                y_sparse = y.to(self.direction_device) # Store on direction_device (CPU if direction_device='cpu')
                if self.clop != 0:
                  old_dirs.append(y_sparse.coalesce()) # Store on direction_device (CPU if direction_device='cpu')
                else:
                  old_dirs.append(y_sparse) # Store on direction_device (CPU if direction_device='cpu')
                if self.clop != 0:
                  old_stps.append(s_sparse.coalesce()) # Store on direction_device (CPU if direction_device='cpu')
                else:
                  old_stps.append(s_sparse) # Store on direction_device (CPU if direction_device='cpu')
#                s_sparse = s.to_sparse().to(self.direction_device) # Store on direction_device (CPU if direction_device='cpu')
#                old_stps.append(s_sparse.coalesce()) # Store on direction_device (CPU if direction_device='cpu')
                ro.append(torch.tensor([(1.0 / ys)], device=self.direction_device)) # Store on direction_device (CPU if direction_device='cpu')

                y_squared = (y * y).sum()
                H_diag = ys / y_squared
                del y_squared
            else:
                H_diag = torch.tensor(1.0).to("cuda") # Default H_diag if ys is too small


            # Prepare history tensors for direction_approximate calculation on calculation device
            old_dirs_calc_device = [h.to("cuda") for h in old_dirs]
            old_stps_calc_device = [s.to("cuda") for s in old_stps]
            ro_calc_device = [r.to("cuda") for r in ro]
            flat_grad_calc_device = flat_grad.to("cuda")
            H_diag_calc_device = H_diag.to("cuda")


            d = self.direction_approximate(old_stps_calc_device, old_dirs_calc_device, ro_calc_device, flat_grad_calc_device, H_diag_calc_device,  clop=self.clop, norm=norm)
            torch.cuda.empty_cache()
            del H_diag

        if prev_flat_grad is None:
            prev_flat_grad = flat_grad
        else:
            prev_flat_grad = flat_grad

        total_norm = torch.linalg.vector_norm(d.coalesce().values(), ord=norm).to("cuda") # Norm on calculation device
        d.div_(total_norm)

        direction_values = d.coalesce().values()
        mask = torch.logical_and(direction_values > -self.clop, direction_values < self.clop)
        direction_values[mask] = 0
        print("direction elements: " + str((direction_values != 0).sum()) + " total: " + str(d.numel()), end=' ')
        indices = d.coalesce().indices()
        valid_indices_mask = direction_values != 0
        valid_indices = indices[:, valid_indices_mask]
        d = torch.sparse_coo_tensor(valid_indices, direction_values[valid_indices_mask], d.size()).coalesce().to(first_param.device) # Move d back to parameter device

        return d, t, H_diag, old_dirs, old_stps, ro, prev_flat_grad, prev_loss
