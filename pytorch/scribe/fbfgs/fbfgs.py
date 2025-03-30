import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
from typing import Optional, Union
import torch
from torch import Tensor
import gc
import psutil
import time

from torch.optim.optimizer import Optimizer, ParamsT

#TODO: add a tensor for unit-indices to save lots of memory
class SparseFlatTensor:
    def __init__(self, starts, ends, values, total_size):
        """
        Represents a 1D sparse tensor using start and end indices for sparse segments.

        Args:
            starts (torch.Tensor): 1D tensor of start indices for each dense segment.
            ends (torch.Tensor): 1D tensor of end indices for each dense segment.
            values (torch.Tensor): 1D tensor containing concatenated values of all dense segments.
            total_size (Tensor): The total size of the 1D tensor.
        """
        self.starts = starts.to(torch.long)
        self.ends = ends.to(torch.long)
        self.values = values # Now a 1D tensor
        self.total_size = torch.tensor(total_size).to(torch.long)

    def __repr__(self):
        return f"SparseFlatTensor(starts={self.starts}, ends={self.ends}, values={self.values}, total_size={self.total_size})"

    def to_dense(self):
        """
        Converts the sparse tensor representation to a dense PyTorch tensor using vectorized operations.
        """
        dense_tensor = torch.zeros(self.total_size, dtype=self.values.dtype, device=self.values.device)
        segment_lengths = (self.ends - self.starts).to(torch.long)
        segment_indices_offsets = torch.repeat_interleave(self.starts.to(torch.long), segment_lengths)
        indices = torch.arange(segment_lengths.sum(), device=self.starts.device).to(torch.long)
        segment_lengths_cumsum = segment_lengths.cumsum(0).to(torch.long)
        start_indices = torch.cat([torch.tensor([0], device=self.starts.device), segment_lengths_cumsum[:-1]]).to(torch.long)
        segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
        segment_internal_indices = (indices - start_indices[segment_ids]).to(torch.long)
        segment_indices = (segment_indices_offsets + segment_internal_indices).to(torch.long)
        dense_tensor[segment_indices] = self.values
        return dense_tensor

    def to(self, device: str):
        """
        Moves all internal tensors to the specified device and returns a new SparseFlatTensor.
        """
        return SparseFlatTensor(
            self.starts.to(device),
            self.ends.to(device),
            self.values.to(device),
            self.total_size.to(device)
        )

    def dot(self, other):
        """
        Computes the dot product of this SparseFlatTensor with another SparseFlatTensor.
        """
        dense_self = self.to_dense()
        dense_other = other.to_dense()
        return torch.dot(dense_self, dense_other)

#    def __add__(self, other):
#        """
#        Element-wise addition of SparseFlatTensor with a dense tensor.
#        """
#        return SparseFlatTensor.add_sparse_dense(self, other)

    def __mul__(self, other):
        """
        Element-wise multiplication of SparseFlatTensor with a dense tensor.
        """
        segment_lengths = (self.ends - self.starts).to(torch.long)
        segment_indices_offsets = torch.repeat_interleave(self.starts.to(torch.long), segment_lengths)
        indices = torch.arange(segment_lengths.sum(), device=self.starts.device).to(torch.long)
        segment_lengths_cumsum = segment_lengths.cumsum(0).to(torch.long)
        start_indices = torch.cat([torch.tensor([0], device=self.starts.device), segment_lengths_cumsum[:-1]]).to(torch.long)
        segment_ids = torch.searchsorted(segment_lengths_cumsum, indices, right=True)
        segment_internal_indices = (indices - start_indices[segment_ids]).to(torch.long)
        segment_indices = (segment_indices_offsets + segment_internal_indices).to(torch.long)

        return SparseFlatTensor(self.starts, self.ends, multiplied_values, self.total_size)

    def rmul(self, scalar):
        """
        Scalar multiplication (right operand) for SparseFlatTensor.
        """
        multiplied_values = self.values * scalar
        return SparseFlatTensor(self.starts, self.ends, multiplied_values, self.total_size)

    def __mul__(self, scalar):
        """
        Scalar multiplication for SparseFlatTensor.
        """
        multiplied_values = self.values * scalar
        return SparseFlatTensor(self.starts, self.ends, multiplied_values, self.total_size)

    @staticmethod
    def add_sparse_dense(sparse_tensor: 'SparseFlatTensor', dense_tensor: Tensor) -> Tensor:
        """
        Adds a SparseFlatTensor to a dense tensor.

        Args:
            sparse_tensor (SparseFlatTensor): The sparse tensor to add.
            dense_tensor (Tensor): The dense tensor to add to.

        Returns:
            Tensor: The dense result of the addition.
        """
        return sparse_tensor.to_dense() + dense_tensor

    @staticmethod
    def sparse_dot_dense(sparse_tensor_arg: 'SparseFlatTensor', dense_tensor):
        """
        Computes the dot product of a SparseFlatTensor with a dense tensor, optimized for sparsity.
        """
        sparse_tensor = sparse_tensor_arg # Explicitly use sparse_tensor_arg
        assert isinstance(sparse_tensor, SparseFlatTensor), "Expected sparse_tensor_arg to be a SparseFlatTensor"
        # Get indices and values from sparse tensor
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

        # Compute dot product
        return torch.dot(sparse_values_from_dense, sparse_tensor.values)


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
        values_local = torch.empty(0, dtype=dtype, device=device)
        total_size_local = torch.tensor(total_size)
    else:
        # Find start and end indices of contiguous segments
        diff = non_zero_indices[1:] - non_zero_indices[:-1]
        segment_ends_indices = torch.nonzero(diff > 1).squeeze() + 1
        segment_starts_indices = torch.cat([torch.tensor([0], device=device), segment_ends_indices])
        segment_ends_indices = torch.cat([segment_ends_indices, torch.tensor([len(non_zero_indices)], device=device)])

        starts_local = non_zero_indices[segment_starts_indices]
        ends_local = non_zero_indices[segment_ends_indices - 1] + 1
        segment_lengths = ends_local - starts_local
        avg_segment_length = segment_lengths.float().mean()
        max_segment_length = segment_lengths.max()
        min_segment_length = segment_lengths.min()
        print(f"Average segment length: {avg_segment_length:.4f}, Max segment length: {max_segment_length}, Min segment length: {min_segment_length}")


        # 1. Generate segment indices without loops - vectorized approach
        segment_indices_offsets = torch.repeat_interleave(starts_local, segment_lengths)

        # 2. Vectorized value extraction using advanced indexing
        values_local = dense_tensor.view(-1)[segment_indices_offsets]
        total_size_local = torch.tensor(total_size)

    return SparseFlatTensor(starts_local, ends_local, values_local, total_size_local)


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


#TODO: cleanup all the AI device mess
#TODO: c3 along with armijo that is c2 but for overconvergence? To prevent early convergence on insta-wolfes? Probably not necessary and would probably slow things down
def _strong_wolfe(
#TODO: c2 = 1 - 1/num_iterations #we always solve given c2 reduction each data point the exact number required
#    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
    obj_func, direction_device, x, t, d, f, g, gtd, c1=1e-20, c2=0.9, tolerance_change=1e-16, max_ls=5, bracket_shift=(1/3), bracket_shove=(1/3), capture_min_step=1e-4, capture_max_step=100
):
#TODO: this irks the mathematician in me.
    if c2 == 0:
      c2 = 0.25
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
#TODO: why don't we scale d by t here, especially since we are normalizing?
    gtd_new_sparse_product = g_new * d
    gtd_new = gtd_new_sparse_product.sum()
    del gtd_new_sparse_product
#    g_new = g_new#
#    gtd_new = gtd_new#
    t_orig = t
    success = False

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    g_prev = g_prev.to(direction_device)
    done = False
    ls_iter = 0

    t_best = t
    t = torch.tensor(t) # Ensure t is a tensor before the loop
    device = gtd.device
#TODO: this can increase loss if f_best is greater than current loss (last iteration loss)
    f_best = torch.tensor(f, device=device)
    g_best = g
    ls_iter=0
    stall_wolfe=0

    while ls_iter < max_ls:
#TODO: we can calculate the delta here for insta wolfes and adjust t by the difference, essentially measuring the drift of the interpolation to see if its shifting left or right to try to stay in the min as long as possible over time
#TODO: e.g.: if wolfe is increasing shift up t, if armijo is increasing, shift down t. We may be able to formulate this as a liner equation or a ratio
        # check conditions

        if (f_new > (f + c1 * t * gtd)) :  # or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
#            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

#TODO: <= for ward condition should be < and just allow first iteration to not check ward condition
        if abs(gtd_new) <= -c2 * gtd: # and f_new <= f_best :
            bracket = [t]
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
            t_prev, f_prev, gtd_prev.to("cuda"), t, f_new, gtd_new.to("cuda"), bounds=(min_step, max_step)
        )
        t = torch.tensor(t) #.item() # get scalar value from tensor

        # next step
        t_prev = tmp
        f_prev = f_new
#        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        g_prev = g_new.to(direction_device)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new_sparse_product = g_new * d
        gtd_new = gtd_new_sparse_product.sum()
        del gtd_new_sparse_product
#        g_new = g_new#
        ls_iter += 1
        #RELAXED WOLFE CONDITION
#        cur_c2 =  abs(gtd_new) - -gtd  #TODO: inverted case
        if f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
          success = True
          stall_wolfe = 0
          t_best = t
          f_best = torch.tensor(f_new, device=device)
          g_best = g_new

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
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.

    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
#    while not done and ls_iter < max_ls:
    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    # WOLFE PACK: find the best strong wolfe point in case we fail to zoom.

    #NOTE: we wait for bracket to collapse, we dont use max linesearch here, if it takes too long turn the bracket hyperparameters up.
    while not done  and ls_iter < max_ls:
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

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0], # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1], # type: ignore[possibly-undefined]
        )
        t = torch.tensor(t)
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
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new_sparse_product = g_new * d
        gtd_new = gtd_new_sparse_product.sum()
        del gtd_new_sparse_product
#        g_new = g_new#
        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?


        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos] : #or f_new > f_best: #NOTE: Ward condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone()  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
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
            if f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
    #          print("---GOT NEW WOLFE PACK---")
              success = True
              stall_wolfe = 0
              t_best = t
              f_best = torch.tensor(f_new, device=device)
              g_best = g_new

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
#            bracket_g[low_pos] = g_new.clone()
            bracket_g[low_pos] = g_new
# type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new
        stall_wolfe += 1
        if stall_wolfe >= 5:
          print("STALL WOLFE")
        if ls_iter >= max_ls: # Return Wolfe pack if max ls reached in zoom phase
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
        tolerance_grad: float = 1e-16,
        tolerance_change: float = 1e-16,
        history_size: int = 2,
        c1: float = 1e-3,
        c2: float = 0.25,
        line_search_fn: Optional[str] = None,
        bracket_shift: float =(1/3),
        bracket_shove: float =(1/3),
        capture_min_step: float =1.,
        capture_max_step: float =100,
        clop: float = 5e-7,
        direction_device: str = 'cuda',
        norm: float = 1.0
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
            norm=norm
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
#TODO: rename
    def _gather_flat_grad(self):
        views = []
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
        grad = torch.cat(views, 0)
#        norm_val = torch.linalg.vector_norm(grad, ord=1.)
#        grad = grad/norm_val
#        return torch.cat(views, 0).to(self.direction_device)
        return grad
#TODO: clip out NaN based on dtype max value
#        return grad_raw #.to(self.direction_device)

    # gather flat grads with L2 Normalization
#TODO: rename
    def _gather_norm_flat_grad(self, norm, isClop = True):
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
#        norm = torch.linalg.vector_norm(views, 2)
#        views.div_(norm)
#TODO: does l1 need a norm scaling parameter or does it naturally scale since it has to sum to one anyways (values that are essentially 0 dont add anything to the norm so it should automatically balance). We may also want a scaling value since large networks might end up clopping too much or even dropping too much with l1. Can we tune this normal scaling value with the same hyperparameter used for clopping s.t. its a hyperparameter that is proportional to a "sub net size"? Probably cant just be one hyperparameter, but can we pass in values 0>x<1? essetially the l0.5 norm for scaling up a bit to account for precision losses? Test this but likely we need a hyperparameter to scale the norm we got from l1.
#TODO: what if we normaling by the max value and let clopping handle what the l1 would do anyways? we would only need to tune the clopping hyperparameter and would get essentially what we want with l1
        #Clop
#TODO: may be worth taking the top K here to have deterministic memory, do this after clopping to create a floor for allocation since we want to allow very sparse outlier gradients
#        if isClop:
#          print("gradient elements: " + str((views != 0).sum()) + " total: " + str(views.numel()), end=' ')
#          views[torch.logical_and(views > -self.clop,views < self.clop)] = 0
#          views = views.to_sparse()
        return views #.to("cpu")
    #TODO: clip out NaN based on dtype max value
    #        return grad_raw #.to("cpu")

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            if update.is_sparse:
                sparse_indices = update.coalesce().indices()
                sparse_values = update.coalesce().values()

                # Extract relevant slice from sparse tensor
                mask = torch.logical_and(sparse_indices[0, :] >= offset, sparse_indices[0, :] < offset + numel)
                view_indices = (sparse_indices[:, mask] - offset).to(p.device) # Adjust indices to be relative to the view
                view_values = sparse_values[mask].to(p.device)
                view = torch.sparse_coo_tensor(view_indices, view_values, torch.Size([numel]), dtype=update.dtype, device=p.device).coalesce() #TODO: verify via profiling if coalesce is necessary

                p_flat = p.view(-1)
                if view_values.numel() > 0:  # Check if there are any values to update
                    index = view_indices[0, :]  # Get the indices for index_add_
                    p_flat.index_add_(0, index.to(p_flat.device), (view_values * torch.tensor(step_size).to(p_flat.device)))  # Use index_add_ for vectorized update


            else: #dense path for non-sparse tensors just in case
                view = update[offset : offset + numel]
                # view as to avoid deprecated pointwise semantics
                p.add_(view.view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

#TODO: we can just clone the bitmask of the sparse gradients since those are the only params we are going to modify
    def _clone_param(self):
#        return [p.clone(memory_format=torch.contiguous_format).to(self.direction_device) for p in self._params]
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
#        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
#        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    def _needle_directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
#        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.jit.script
    def direction_approximate(old_stps: list[SparseFlatTensor], old_dirs: list[SparseFlatTensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, direction_device: str,t: float, clop: float, norm: float) -> Tensor:
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
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=direction_device)

        for i in range(num_old - 1, -1, -1):
            #direction_similarity = (old_dirs[i].to_dense().to("cuda") * q).sum().item() # Convert to dense here
            sparse_dir_i = old_dirs[i] # Explicitly get old_dirs[i]
            direction_similarity = SparseFlatTensor.sparse_dot_dense(sparse_dir_i.to("cuda"), q).item() # Use sparse_dir_i
            aligned = direction_similarity >= similarity  or direction_similarity <= -similarity
            direction_alignment_mask[i] = aligned
            dense_old_dir = torch.zeros_like(q) # Initialize dense_old_dir as a zero tensor
            if direction_alignment_mask[i]:
              al[i] = direction_similarity * ro[i].item() # Use direction_similarity which is now computed with SparseFlatTensor
              sparse_old_dir_scaled = old_dirs[i].to("cuda") * ((-al[i])) # Scale sparse tensor
              q = SparseFlatTensor.add_sparse_dense(sparse_old_dir_scaled.to("cuda"), q) # Sparse addition
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

        d = q.mul(H_diag)
        be_i = torch.empty_like(d, dtype=q.dtype, device="cuda") # Preallocate be_i for second loop
        del q

#TODO: vectorize alignment mask here since its immutable
        for i in range(num_old):
            dense_old_dir = torch.zeros_like(d) # Initialize dense_old_dir here
            if direction_alignment_mask[i]:
              #be_i.copy_((old_dirs[i].to_dense().to("cuda") * d).to_dense()) # Convert to dense here
#              dense_old_dir = old_dirs[i].to_dense().to("cuda")
              be_i.copy_((old_dirs[i].to_dense().to("cuda") * d).to_dense())
              # del dense_old_dir # DEL 11: Initialize dense_old_dir before if block in second loop
              # d.add_(old_stps[i].to_dense().to("cuda"), alpha=al[i] - be_i.sum() * ro[i].item()) # Convert to dense here
              alpha_val = al[i] - be_i.sum() * ro[i].item()
              sparse_old_stp_scaled = old_stps[i].to("cuda") * (alpha_val) # Scale sparse tensor
              d = SparseFlatTensor.add_sparse_dense(sparse_old_stp_scaled.to("cuda"), d) # Sparse addition
              #del dense_old_stp


        print(hit_miss)
#TODO: we may increase efficacy and reduce tearing by supplemnting clopping with a lower order norm
        total_norm = torch.linalg.vector_norm(d, ord=norm).to("cuda")
        d = d.div_(total_norm)
#        direction = d
#        mask = torch.logical_and(direction > -clop, direction < clop) #TODO: extract to sub_variance hyperparameter
#        direction[mask] = 0
#        d = direction.to_sparse()
#        print("direction elements: " + str((direction != 0).sum()) )
#        del mask # DEL 9: mask is no longer needed
        return d

    @torch.jit.script
    def dense_direction_approximate(old_stps: list[Tensor], old_dirs: list[Tensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, direction_device: str,t: float, clop: float, norm: float) -> Tensor:
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
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=direction_device)

        for i in range(num_old - 1, -1, -1):
            direction_similarity = (old_dirs[i].to("cuda") * q).sum().item() # Convert to dense here
            aligned = direction_similarity >= similarity  or direction_similarity <= -similarity
            direction_alignment_mask[i] = aligned
            if direction_alignment_mask[i]:
              al[i] = direction_similarity * ro[i].item() # Use direction_similarity which is now computed with SparseFlatTensor
              q = q + (old_dirs[i].to("cuda") * ((-al[i]))) # Dense addition
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

        d = q.mul(H_diag)
        be_i = torch.empty_like(d, dtype=q.dtype, device="cuda") # Preallocate be_i for second loop
        del q

#TODO: vectorize alignment mask here since its immutable
        for i in range(num_old):
            if direction_alignment_mask[i]:
              be_i.copy_((old_dirs[i].to("cuda") * d))
              alpha_val = al[i] - be_i.sum() * ro[i].item()
              d = d + (old_stps[i].to("cuda") * (alpha_val)) # Dense addition


        print(hit_miss)
#TODO: we may increase efficacy and reduce tearing by supplemnting clopping with a lower order norm
        total_norm = torch.linalg.vector_norm(d, ord=norm).to("cuda")
        d = d.div_(total_norm)
#        direction = d
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

      # NOTE: FBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      state = self.state[self._params[0]]
#      state.setdefault("func_evals", 0)
#      state.setdefault("n_iter", 0)
#
      # evaluate initial f(x) and df/dx
      orig_loss = closure()
      loss = float(orig_loss)
      current_evals = 1
#      state["func_evals"] += 1
      al = []

#      flat_grad = self._gather_flat_grad()
#      flat_grad = self._gather_norm_flat_grad(2, False)
#      flat_grad = self._gather_flat_grad()
#TODO: remove this if we remove gradient normalization.
#      opt_cond = flat_grad.abs().max() <= tolerance_grad #TODO: see TODO below. Can this ever happen with normalization? shouldn't.
#      opt_cond = flat_grad.abs().max() <= 0 #TODO: see TODO below. Can this ever happen with normalization? shouldn't.

#TODO: HARDCORE.
      # optimal condition
#      if opt_cond :#or loss.isnan:# NOTE: this is a NaN check via equivalence
#          print("GRAD CONVERGED") #TODO: if we throw out the hessian, will the gradient norm be able to fix this? No, the normalization scalar coeficient is clamped @ 1 so we only scale the norm down.
						#TODO: can we flip the c2 condition to force curvature to escape like momentum?or like a cosine schedule of learning rate based on sub-optimal convergence? ideally we just set c2 correctly but this would be much more robust and easier to tune.
#TODO: instead of resetting, or alongside resetting, flip the linesearch to search for > C2 condition as a momentum factor.
#          print("RESET")
#          d = flat_grad.neg()
#          old_dirs = []
#          old_stps = []
#          ro = []
#          H_diag = 1
#          return orig_loss

#TODO: put old_dirs, steps and ro on self.direction_device. Perform the direction calculation as efficiently as possible with this constraint so we can use main memory for history size
      # tensors cached in state (for tracing)
#      d = state.get("d")
#      t = state.get("t")
#      old_dirs= []
#      old_stps= []
#      ro= []
#TODO: initialize al here not itl
#TODO: configure: keep_hessian, grad_norm, fragment_sub_variance, direction_norm -- hyperparameters for L-BFGS-NS (reset hessian per datapoint/linesearch failure, sub_variance for fragmentation dropout, grad/direction (L1/L2)
#TODO: also expose C1 and C2, we would expose max_linesearch but instead expose stall_wolfe since its a more informed and as reliable heuristic metric
      if "old_dirs" in state:
        old_dirs = state.get("old_dirs")
        old_stps = state.get("old_stps")
        ro = state.get("ro")
#TODO: TEST
#      H_diag = state.get("H_diag")
#      prev_loss = state.get("prev_loss")
#TODO: this may leak when we reset and assign prev_flat_grad to None
        prev_flat_grad = state.get("prev_flat_grad")
      else:
        old_dirs= []
        old_stps= []
        ro= []
        prev_flat_grad = None

      n_iter = 0
#      d = flat_grad.neg() # Initialize d on direction_device
      first_param = next(self.param_groups[0]['params'].__iter__())
      t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device)
      ls_failed = False
      # optimize for a max of max_iter iterations
#TODO: we arent storing the last iteration in history. Consider reworking the last iteration logic for step function
#      while n_iter < max_iter:
      while True:
          # keep track of nb of iterations
          gc.collect()
          n_iter += 1
          print("[CRAM]")

          ############################################################
          # compute gradient descent direction
          ############################################################
          #TODO: DEPRECATED, the reset logic should be extracted, this should just be initializing d as grad etc.
#TODO: or if history is empty. Better if we do this by history in case we reset the approximation.
#          if prev_flat_grad is None :
          if n_iter == 1 or prev_flat_grad is None:
              restart = False
#TODO: use the proper flat_grad (the l1 instead of l2) here since we dont calculate direction first
              print("RESET")
              d = self._gather_flat_grad().neg()
              flat_grad = self._gather_flat_grad()
#TODO: if we do this we should norm inf for Rollover stability
              total_norm = torch.linalg.vector_norm(d, ord=norm) # Move total_norm to direction_device
              d = d/total_norm
#              d[torch.logical_and(d > -self.clop,d < self.clop)] = 0
#              d = d.to_sparse()
              H_diag = 1
              t = 1
              gc.collect()
#              print("d elements: " + str((d.values() != 0).sum()) )
          else:
              torch.cuda.empty_cache() # Clear cache before direction calculation
              if prev_flat_grad is not None:
                  prev_flat_grad = prev_flat_grad # Move prev_flat_grad to direction_device
#TODO: ensure this is on GPU
              y_dense = flat_grad.to("cuda").sub(prev_flat_grad.to("cuda"))
              s_dense = (d.mul(t)) # Define s_dense here
              ys = y_dense.dot(s_dense) # Calculate ys here after s is SparseFlatTensor
#Clop
#TODO: can we scale after norm to prevent the magnitude after clopping from being epsilon? I think this would be mathematically unstable but would help with the direction approximation's curvature
#TODO: essentially, scale the result of the clop s.t. the max value is 1. Would this just be the inf ord?
              total_norm_y = torch.linalg.vector_norm(y_dense, ord=norm) # Move total_norm to direction_device
              y_dense = y_dense/total_norm_y
              y_dense[torch.logical_and(y_dense > -self.clop,y_dense < self.clop)] = 0
              total_norm_s = torch.linalg.vector_norm(s_dense, ord=norm) # Move total_norm to direction_device
              s_dense = s_dense/total_norm_s
              s_dense[torch.logical_and(s_dense > -self.clop,s_dense < self.clop)] = 0
              y = dense_to_sparse_flat_tensor(y_dense)
              s = dense_to_sparse_flat_tensor(s_dense)

#              y = y*total_norm
#
#              total_norm = torch.linalg.vector_norm(y, ord=float("inf")) # Move total_norm to direction_device
#              y = y/total_norm
#              if self.clop != 0:
#                y = y.to_sparse()
              print("d-delta elements: " + str((d.to_dense() != 0).sum()) + " total: " + str(d.to_dense().numel()), end=' ')
              print("S elements: " + str((s_dense != 0).sum()) + " total: " + str(s_dense.numel()), end=' ') # s_dense is still dense here
              print("y-delta elements: " + str((y.to_dense() != 0).sum()) + " total: " + str(y.to_dense().numel()), end=' ')
#TODO: may need to calculate ys before
#              ys_sparse_product = y * s
#              ys = ys_sparse_product.sum()#y*s
#              del ys_sparse_product
#TODO: SCALE HESSIAN^-1 COMPONENTS BY ERROR TO REFINE APPROXIMATION MORE EFFICIENTLY
#TODO: with normalization, armijo should be able to solve s.t. c1 <= 1 since loss reduction is 1:1 if the direction approx is 100% accurate since direction is normalized. We also can expect flat_grad.dot(d) to be 0 if approx is 100% accurate since we set number of iterations based on c2 condition convergence minima. e.g.: c2 = 0.9 we do 10 iterations for 100% reduction.
		#TODO: ys = flat_grad.dot(d)  * ys ? #TODO: (abs(gtd_prev) - -gtd ) * ys TODO: which  of these is better? they both make sense to me right now
#              if ys > set this to 1e-10: #TODO:  this may not work with normalized unit vector failsafe. 1e-16 or precision of assigned dtype or better yet ys > 0
#              if ys > 1e-16:
#TODO: double check the math to ensure this will account for opposing direction-curvature
#              if  ys >= 1e-8 or ys <= -1e-8:
#TODO: ys is extremely important. This determines how quickly we can discover otherwise zero partials and therefore move the direction via history. It may be worth using double precision throughout this optimizer to account for the otherwise instability | ys -> epsilon
#TODO: if the case where ys <= -N is correct, can we find contradictions in the direction wrt curvature s.t. we can remove 2 indices in the approximation if they are sufficiently aligned and opposing? this may prevent us from popping valuable directions while maintaining directions that have been disproved
#TODO: we can keep the gradient in whatever precision the closure gives us then upscale to the direction in 64 etc. (configurable hyperparam?). This should scale s.t. the precision greatly accounts for sparsity (more memory efficient to use higher precision?)
              if  ys >= 1e-8  or ys <= -1e-8:
                # updating memory
#                if len(old_dirs) <= history_size:
#TODO: fix this so any cuda device gets this
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
#                if self.clop > 0:
#                  y_sparse = y.to_sparse()
#                if self.clop > 0:
#                  y_sparse = y.to(self.direction_device) # Store y_sparse on direction_device
#                  old_dirs.append(y_sparse.coalesce().to(self.direction_device)) # NOTE: was cpu
#                else:
#                  y_sparse = y.to(self.direction_device) # Store y_sparse on direction_device
#                  old_dirs.append(y_sparse.to(self.direction_device)) # NOTE: was cpu
#                if self.clop > 0:
#                  s_sparse = s.to_sparse().to(self.direction_device).to(self.direction_device) # Store s_sparse on direction_device
#                  old_stps.append(s_sparse.coalesce().to(self.direction_device)) # NOTE: was cpu
#                else:
#                  s_sparse = s.to(self.direction_device).to(self.direction_device) # Store s_sparse on direction_device
#                  old_stps.append(s_sparse.to(self.direction_device)) # NOTE: was cpu
                old_dirs.append(y.to(self.direction_device)) # Store y as SparseFlatTensor
                old_stps.append(s.to(self.direction_device)) # Store s as SparseFlatTensor
                ro.append(torch.tensor([(1.0 / ys)], device=self.direction_device)) # NOTE: was cpu #TODO: can we include information on convergence here. This may be an observation of the approximation accuracy. Also consider the alignment (gtd being as close to zero as possible). essentially we would be scaling how much the approximation is influenced by an entry based on its ability to converge.
              if n_iter > max_iter:
                break
#TODO: break here on n_iters
              # update scale of initial Hessian approximation
#TODO: was this also shifted? check the original implementation
              y_squared = y_dense.dot(y_dense)
              H_diag = ys / y_squared  # (y*y)
              del y_squared


              y = y #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              ys = ys #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.

              # compute the approximate (L-BFGS) inverse Hessian
              # multiplied by the gradient
              num_old = len(old_dirs)

#              if "al" not in state:
#                state["al"] = [None] * history_size
#              al = [None] * history_size
              al = [None] * num_old
#              al = state["al"]

              # iteration in L-BFGS loop collapsed to use just one buffer
#              q = flat_grad.neg()  # Move q to direction_device

              # Move history to direction_device
#              old_dirs_cuda = [tensor.to(self.direction_device) for tensor in old_dirs]
#              old_stps_cuda = [tensor.to(self.direction_device) for tensor in old_stps]
#              ro_cuda = [tensor.to(self.direction_device) for tensor in ro]

              gc.collect()
#              d = self.direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device="cpu", clop=self.clop, clop=self.clop)
#TODO: TEST: use the l1 norm to bootstrap alignment/selection and rely on the l2 for convergence metrics and curvature.
              d = self.direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device="cpu", t=t,  clop=self.clop, norm=norm)

              # Move history back to CPU
#              old_dirs = [tensor.to('cpu') for tensor in old_dirs_cuda]
#              old_stps = [tensor.to('cpu') for tensor in old_stps_cuda]
#              ro = [tensor.to('cpu') for tensor in ro_cuda]

              torch.cuda.empty_cache()

              del H_diag  # DEL 6: H_diag is no longer needed
              # del sparse_product_al # Delete after loop
              # del intermediate_be # Delete after loop

          if prev_flat_grad is None : #or state["n_iter"] == 1:
#              prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)#NOTE: was self.direction_device
              prev_flat_grad = flat_grad#NOTE: was self.direction_device
          else:
#              prev_flat_grad.copy_(flat_grad)#NOTE: was self.direction_device
              prev_flat_grad = flat_grad#NOTE: was self.direction_device
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
          gtd_sparse_product = flat_grad.to("cuda") * d
          gtd = gtd_sparse_product.sum() # g * d
          del gtd_sparse_product
          prev_flat_grad = prev_flat_grad.to(self.direction_device)
#          if state["n_iter"] != 1:
#          avg = gtd.abs().mean()
#          print("got avg: " + str(avg))
##          t = min(1e16, 1/avg)
          t = self.t #TODO: this should be set based on an average of step sizes or something. We can track what the learning rate should be to increase the speed of bracket search without missing points at lower step sizes.
##            t = min(5e5, 5e-5/ avg)
#          print("got t: " + str(t))

#          flat_grad = flat_grad#
#          gtd=gtd#
#          flat_grad = flat_grad#
#          gtd=gtd
#          d = d#
#          d = d
#          t = t#

          # directional derivative is below tolerance
#NOTE: if we dont break here we are surely going to zoom on the bracket. This is preferable to just skipping until the data point aligns with the hessian but may prefer reseting the hessian instead.
#          if gtd > -tolerance_change:
#              break

          # optional line search: user function
          ls_func_evals = 0
          if line_search_fn is not None:
              # perform line search, using user function
              if line_search_fn != "strong_wolfe":
                  raise RuntimeError("only 'strong_wolfe' is supported")
              else:
                  x_init = self._clone_param()

                  def obj_func(x, t, d):
                      return self._directional_evaluate(closure, x, t, d)

                  gc.collect()
                  success, loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                      obj_func, self.direction_device, x_init, t, d, loss, flat_grad, gtd, c2=c2,c1=c1, bracket_shift=bracket_shift, bracket_shove=bracket_shove, capture_min_step=capture_min_step, capture_max_step=capture_max_step
                  )
#                      obj_func, x_init, t, d, loss, flat_grad, gtd, c2=(1-1/max_iter)
#TODO: Another solution is to decrease the norm order if the loss doesnt reduce and only break when decreasing the order does not also decrease the loss
              Needle = False
              if not success: #TODO: we chase misprinted lines
                if  ls_failed: #TODO: we chase misprinted lines
                  t = 1. #Reset t to 1 for after needling
                  best_needle_loss = float(1.) # Initialize best_needle_loss here to ensure it's always defined
                  print("saddle-search subroutine..")
                  Needle = True
                  first_param = next(self.param_groups[0]['params'].__iter__())
                  needle_t = torch.tensor(1.0, dtype=first_param.dtype, device=first_param.device) #Unit vector until we restore curvature
                  best_needle_t = needle_t.clone()
#TODO: just use x_init
#TODO we need to check convergence on the needle
                  x_init_needle = self._clone_param() # Clone params for needle search

                  # Iteratively increase t until loss no longer decreases
#TODO: initialize the gtd here instead of the original gtd.
                  flat_grad = self._gather_flat_grad()
                  d_needle = flat_grad.neg()
                  total_norm = torch.linalg.vector_norm(d_needle, ord=1/3)
                  d_needle = d_needle.div_(total_norm)
#NOTE: I use l1 norm so we converge quickly and dont over apply the needle
                  gtd = d_needle * flat_grad
                  gtd = gtd.sum()
                  while True:
#TODO: use raw gradients so we dont double norm here
#TODO: only calculate the d_needle once since we are linesearching it
#TODO: sharpen routine first before pressure
                      current_needle_loss, flat_grad= self._needle_directional_evaluate(closure, x_init_needle, needle_t, d_needle) # Use directional_evaluate
                      gtd_needle_sparse_product = flat_grad * d_needle #TODO: use raw gradients so we dont double norm here
                      gtd_needle = gtd_needle_sparse_product.sum() # g * d
                      del gtd_needle_sparse_product
                      armijo_condition = current_needle_loss <= best_needle_loss + c1 * needle_t * gtd
                      if current_needle_loss <= best_needle_loss and abs(gtd_needle) > -c2 * gtd and armijo_condition: #and abs(gtd_needle) <= -c2 * gtd: #abs(gtd_new) <= -c2 * gtd:
                          best_needle_loss = current_needle_loss
                          best_needle_t = needle_t.clone()
#TODO: try a exponential scaling here of 2**n. exponential may not be right but we need something more efficient like interpolation. we may be able to just guess how far by the convergence rate
                          needle_t = 2*needle_t  # Increase t for next iteration
                      else:
                          break # Stop if loss no longer decreasing


                  self._add_grad(best_needle_t, d_needle) # Use best t for add_grad

                  loss_device = d.device
                  print(f" \n -----------got needle stepsize: {needle_t} and loss: \033[92m{current_needle_loss}\033[0m on device: {loss_device}-----------") # Use best_needle_loss
                  prev_flat_grad = None

                print("\033[91mLinesearch failure, resetting..\033[0m")
                ls_failed = True
              else:
                ls_failed = False

#TODO: I dont like having to do this but we want l2 for the direction selection.
#TODO: dont reset the Hessian if we are using prev step size since one iteration may be insufficient to bracket down
#                if "old_dirs" in state:
#                  state["old_dirs"].clear()
#                  state["old_stps"].clear()
#                  state["ro"].clear()
#TODO: dont clear these? may leak here
#                old_dirs = []
#                old_stps = []
#                ro = []
#                state["n_iter"] = 0
#              flat_grad = flat_grad.to("cuda")
              if  ls_failed and Needle == False: #and Needle == False: 
                flat_grad = prev_flat_grad
                prev_flat_grad = None
              else:
                self.t  = t
          if not ls_failed:
            first_param = next(self.param_groups[0]['params'].__iter__())
            t = t.to(first_param.device)
            d = d.to(first_param.device)
            self._add_grad(t, d)
            loss_device = d.device
            print(f" \n -----------got stepsize: {t} and loss: \033[92m{loss}\033[0m on device: {loss_device}-----------") # Use best_needle_loss
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
          if n_iter == max_iter or loss == 0:
              break

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
            "ro":  state_dict.get("ro", []),
            "prev_flat_grad": state_dict.get("prev_flat_grad", None),
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
            state = self.state[self._params[0]]
            device = self.direction_device # Get the device of the model parameters
            state["old_dirs"] = [tensor.to(device) for tensor in history.get("old_dirs", [])] # Load history and move to direction_device
            state["old_stps"] = [tensor.to(device) for tensor in history.get("old_stps", [])] # Load history and move to direction_device
            state["ro"] = [tensor.to(device) for tensor in history.get("ro", [])] # Load history and move to direction_device
            state["prev_flat_grad"] = history.get("prev_flat_grad", None) # Load history
            t_val = history.get("t", 1) # Load step size t, default to 1 if not found
            if isinstance(t_val, torch.Tensor):
                self.t = t_val.item()
            else:
                self.t = t_val
            state["n_iter"] = history.get("n_iter", 0) # Load iteration count n_iter, default to 0 if not found

            if state["prev_flat_grad"] is not None:
                state["prev_flat_grad"] = state["prev_flat_grad"].to(device) # Move prev_flat_grad to direction_device if it exists
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
