#In Memory of Oshkosh, my pet Dalmatian.
import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'
from typing import Dict, List, Optional, Tuple, Union
from torch import device
import torch
from torch import Tensor
import torch.nn.utils
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
from .sparse_flat_tensor import SparseFlatTensor
__all__ = ["FBFGS"]
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
    g = g
    t = torch.tensor(1) # Ensure t is a tensor before the loop
    f_new, g_new = obj_func(t, d)
    ls_func_evals = 1
    gtd_new = (g_new * d).sum() # Keep as scalar tensor
    success = False
    is_nan = False
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0 # Initialize ls_iter
    t = torch.tensor(1) # Ensure t is a tensor before the loop
    t_best = t
    device = gtd.device
    f_best = torch.tensor(f, device=device)
    g_best = g
    gtd_best = gtd
    if f_new < f_best  and done != True  and f_new == f_new and f_new <= (f - abs(c1 * t * gtd)) :
      success = True
      stall_wolfe = 0
      t_best = t
      f_best = torch.tensor(f_new, device=device)
      g_best = g_new
      gtd_best = gtd_new
    gc.collect()
    ls_iter=0
    stall_wolfe=0
    while ls_iter < max_ls:
        if ( abs(gtd_new) <= -c2 * gtd and f_new < f) or (f_new < (f + c1 * t * gtd) ):
            bracket = [t]  #type: ignore[list-item]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            success = True
            t_best = t
            f_best = torch.tensor(f_new, device=device)
            g_best = g_new
            print("FAST WOLFE")
            break
        if gtd_new >= 0 or True:
            print("NOT DESCENDING")
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        min_step = t + capture_min_step * (t - t_prev)#TODO: this can miss, if t+0.01 breaks both armijo and wolfe condition (the interpolation is steep)
        lower_bracket = min(t_prev, t)
        upper_bracket = max(t_prev, t)
        max_step = upper_bracket * capture_max_step
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        if f_new != f_new:  #Check for NaN
          is_nan = True
        t = torch.tensor(t, device=device) # Ensure t is a tensor on the correct device
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.to(direction_device)
        gtd_prev = gtd_new # type: ignore[assignment] # type: ignore[assignment]
        f_new, g_new = obj_func(t, d)
        ls_func_evals += 1 # Increment func evals after new evaluation
        gtd_new = (g_new * d).sum() # Keep as scalar tensor
        ls_iter += 1
        if f_new < f_best  and done != True and f_new == f_new :
          success = True
          stall_wolfe = 0
          t_best = t
          f_best = torch.tensor(f_new, device=device)
          g_best = g_new
          gtd_best = gtd_new
    if ls_iter == max_ls:
        bracket = [t_prev, t]
        bracket_f = [f_prev, f_new]
        bracket_g = [g_prev, g_new]
        bracket_gtd = [gtd_prev, gtd_new]
    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    while not done  and ls_iter < max_ls and not is_nan:
        print("zooming..")
        if abs(bracket[1] - bracket[0])  < tolerance_change  : # or stall_wolfe >= 5 :
           print("WOLFE PACK")
           return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals
        t_prev = t
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0], # type: ignore[possibly-undefined]
            bracket_gtd[0],
            bracket[1],
            bracket_f[1], # type: ignore[possibly-undefined]
            bracket_gtd[1],
        )
        t = torch.tensor(t, device=device) # Ensure t is a tensor on the correct device
        if f_new != f_new:
          t = torch.tensor(1.)
          is_nan = True
        eps = bracket_shift * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
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
        f_new, g_new = obj_func(t, d) # Single evaluation
        ls_func_evals += 1 # Increment func evals
        gtd_prev = gtd_new
        gtd_new = (g_new * d).sum() # Keep as scalar tensor
        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?
        if f_new < f_best and f_new == f_new and f_new <= (f + c1 * t * gtd):
          success = True
          stall_wolfe = 0
          t_best = t
          f_best = torch.tensor(f_new, device=device)
          g_best = g_new
        print("Ward condition: " + str((gtd_new + gtd_prev)/(f_new - f_prev) ))
        if f_new > (f + c1 * t * gtd)  or f_new >= bracket_f[low_pos] or f_new != f_new: #or f_new > f_best: #NOTE: Ward condition#NOTE: PREV SETTING
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0) # type: ignore[possibly-undefined]
        else:
            if abs(gtd_new) <= -c2 * gtd and f_new < f_best : 
                print("STRONG WOLFE")
                success = True
                done = True
                t_best = t
                f_best = torch.tensor(f_new, device=device)
                g_best = g_new.to(direction_device)
                break
            elif gtd_new * (bracket[high_pos] - bracket[low_pos])>= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
#            if abs(gtd_new) < abs(gtd_best) and f_new == f_new :
#              success = True
#              stall_wolfe = 0
#              t_best = t
#              f_best = torch.tensor(f_new, device=device)
#              g_best = g_new
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new
            bracket_gtd[low_pos] = gtd_new
        stall_wolfe += 1
        if ls_iter >= max_ls and done != True and success == True: # Return Wolfe pack if max ls reached in zoom phase
          print("WOLFE PACK MAX LS")
          return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals
    return success, f_best, g_best.to(optimizer_device), t_best, ls_func_evals
@torch.jit.script
def _split_tensor_to_groups_jit(
    flat_tensor: torch.Tensor,
    offsets: List[int],
    norm_group: Optional[Union[int, float]]
) -> List[torch.Tensor]:
    num_layers = len(offsets)
    param_sizes: List[int] = []
    for k in range(num_layers):
        if k + 1 < num_layers:
            param_sizes.append(offsets[k+1] - offsets[k])
        else:
            param_sizes.append(flat_tensor.numel() - offsets[k])
    total_size = flat_tensor.numel()
    
    # Handle norm_group == 0 as norm_group == 1
    effective_norm_group: float = 1.0 # Initialize with a default value
    if norm_group is None:
        effective_norm_group = 1.0
    elif norm_group == 0 or norm_group == 0.0:
        effective_norm_group = 1.0
    else:
        effective_norm_group = float(norm_group)
    # Determine number of groups
    groups: List[torch.Tensor] = []
    current_offset = 0
    if effective_norm_group == 1.0: # One group per layer (matrix norm)
        for size in param_sizes:
            groups.append(flat_tensor[current_offset:current_offset+size])
            current_offset += size
        return groups
    elif effective_norm_group == float(num_layers): # One group for all parameters (vector norm)
        return [flat_tensor]
    elif effective_norm_group < 1.0: # Split each layer into int(1/norm_group) groups
        groups_per_layer = int(1.0 / effective_norm_group)
        for layer_idx in range(num_layers):
            size = param_sizes[layer_idx]
            layer_tensor = flat_tensor[current_offset:current_offset+size]
            # Split this layer into groups_per_layer groups
            min_size = size // groups_per_layer
            remainder = size % groups_per_layer
            split_sizes: List[int] = []
            for i in range(groups_per_layer):
                split_size = min_size
                if i < remainder:
                    split_size += 1
                split_sizes.append(split_size)
            
            layer_groups = torch.split(layer_tensor, split_sizes, dim=0)
            for lg in layer_groups:
                groups.append(lg)
            current_offset += size
        return groups
    else: # norm_group > num_layers: Split total parameters into norm_group groups
        num_groups = int(effective_norm_group)
        
        group_size = total_size // num_groups
        remainder = total_size % num_groups
        
        for i in range(num_groups):
            current_group_size = group_size
            if i < remainder:
                current_group_size += 1
            groups.append(flat_tensor[current_offset:current_offset+current_group_size])
            current_offset += current_group_size
        return groups
@torch.jit.script
def _apply_backward_loop_update(
    q: torch.Tensor,
    sparse_dir_i: SparseFlatTensor,
    sparse_stp_i: SparseFlatTensor,
    y_norms: List[torch.Tensor],
    i: int,
    orthogonality: float,
    al: torch.Tensor,
    direction_alignment_mask: torch.Tensor,
    direction_similarities: List[float],
    optimizer_device: str,
    ro_i: torch.Tensor,
    q_inv_norm: float,
    offsets: List[int],
    radius_ball: float,
    norm_group_y: Optional[Union[int, float]],
) -> Tuple[torch.Tensor, torch.Tensor, List[float], float]:
    inv_dir_norm = torch.tensor(y_norms[i].item() )
    normalized_dir = sparse_dir_i * inv_dir_norm
    # Use the passed q_inv_norm directly
    direction_similarity = SparseFlatTensor.sparse_dot_dense(normalized_dir, q * q_inv_norm).item()
    aligned =  -orthogonality <= direction_similarity <= orthogonality
    direction_alignment_mask[i] = aligned
    direction_similarities.append(direction_similarity)
    if direction_alignment_mask[i]:
        alpha = SparseFlatTensor.sparse_dot_dense(sparse_stp_i, q).item()
        al[i] = alpha * ro_i.item()
        sparse_old_dir_scaled = sparse_dir_i * torch.tensor(-al[i])
        q = SparseFlatTensor.add_sparse_dense(sparse_old_dir_scaled, q) # Update q
        # Apply the norm_select logic directly within the jitted function
        # Split tensor into groups based on norm_group_y
        chunks = _split_tensor_to_groups_jit(q, offsets, norm_group_y)
        # Vectorized and simplified ball projection (radius_ball is assumed > 0)
        # Handle empty chunks by replacing them with a tensor that won't affect calculations
        processed_chunks = [chk if chk.numel() > 0 else torch.tensor(0., device=q.device) for chk in chunks]
        # Calculate L2 norms for all chunks at once
        l2_norms = torch.stack([torch.linalg.vector_norm(chk, ord=2) for chk in processed_chunks])
        # Calculate factors: Only scale down when l2_norm/radius_ball >= 1
        # Use torch.where for vectorized conditional logic
        factors = torch.where(
            (l2_norms / radius_ball >= 1),
            l2_norms / radius_ball,
            torch.tensor(1.0, device=q.device)
        )
        # Apply factors to chunks using a list comprehension (can't directly vectorize across list of tensors)
        chunks = [
            chk / factor
            if chk.numel() > 0 else chk
            for chk, factor in zip(chunks, factors)
        ]
#        if len(chunks) == 0:
#            q = q # No change if no chunks
#        else:
        q = torch.cat(chunks)
        new_q_inv_norm = 1.0 / (torch.linalg.vector_norm(q, ord=2).item() )
        return q, direction_alignment_mask, direction_similarities, new_q_inv_norm # Return new_q_inv_norm
    
    return q, direction_alignment_mask, direction_similarities, q_inv_norm # Return original q_inv_norm
@torch.jit.script
def _apply_forward_loop_update(
    d: torch.Tensor,
    stp_device: SparseFlatTensor,
    dir_device: SparseFlatTensor,
    al: torch.Tensor,
    idx: int,
    ro_idx: torch.Tensor,
    offsets: List[int],
    radius_ball_s: float,
    norm_group_s: Optional[Union[int, float]]
) -> torch.Tensor:
    dot_product_val = SparseFlatTensor.sparse_dot_dense(dir_device, d)
    alpha_val = al[idx] - dot_product_val * ro_idx.item()
    
    scaled_stp = SparseFlatTensor(
        stp_device.starts, stp_device.ends, stp_device.values * (alpha_val),
        stp_device.total_size, stp_device.unit_indices,
        stp_device.unit_values * (alpha_val) if stp_device.unit_values.numel() > 0 else torch.empty(0, dtype=torch.float32, device=stp_device.values.device)
    )
    d = SparseFlatTensor.add_sparse_dense(scaled_stp, d)
    # Apply the norm_select logic directly within the jitted function (forward loop)
    # Split tensor into groups based on norm_group_s
    d_chunks = _split_tensor_to_groups_jit(d, offsets, norm_group_s)
    processed_d_chunks = [chk_d if chk_d.numel() > 0 else torch.tensor(0., device=d.device) for chk_d in d_chunks]
    # Ball projection (radius_ball_s is assumed > 0)
    l2_norms_d = torch.stack([torch.linalg.vector_norm(chk_d, ord=2) for chk_d in processed_d_chunks])
    
    factors_d = torch.where(
        (l2_norms_d / radius_ball_s >= 1), # Removed epsilon check as requested
        l2_norms_d / radius_ball_s, # Use radius_ball_s directly
        torch.tensor(1.0, device=d.device)
    )
    
    d_chunks = [
        chk_d / factor_d
        if chk_d.numel() > 0 else chk_d
        for chk_d, factor_d in zip(d_chunks, factors_d)
    ]
    
#    if len(d_chunks) == 0:
#        d = d
#    else:
    d = torch.cat(d_chunks)
    return d
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
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
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
        max_iter: int = 20,
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
        radius_s: float = 0,
        radius_y: float = 0,
        radius_ball: float = 0,
        radius_ball_s: float = 1.0,
        direction_device: str = 'cpu',
        optimizer_device: str = 'cuda',
        norm: float = 1.0,
        y_norm: Optional[float] = None,
        rho_rewind: int = 10,
        orthogonality: float = 1e-2,
        max_ls: int = 10,
        prefetch_buffer: int = 20_000_000,
        norm_group_s: Optional[Union[int, float]] = None,  # New parameter
        norm_group_y: Optional[Union[int, float]] = None,  # New parameter
        ro_threshold_rate: float = 1,  # New parameter
        lambda_reg: float = 0.001,  # Regularization strength
    ):
        self.lambda_reg = lambda_reg  # Regularization strength hyperparameter
#        self._last_penalty = torch.tensor(0.0, requires_grad=True)  # Track regularization penalty
        self._last_penalty = torch.zeros(1).to(optimizer_device)
        defaults = dict(
            max_iter=max_iter,
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
            radius_s=radius_s,
            radius_y=radius_y,
            radius_ball=radius_ball,
            radius_ball_s=radius_ball_s,
            direction_device=direction_device,
            optimizer_device=optimizer_device,
            norm=norm,
            y_norm=y_norm,
            rho_rewind=rho_rewind,
            orthogonality=orthogonality,
            max_ls=max_ls,
            prefetch_buffer=prefetch_buffer,
            norm_group_s=norm_group_s,  # Add to defaults
            norm_group_y=norm_group_y,  # Add to defaults
            ro_threshold_rate=ro_threshold_rate,  # Add to defaults
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "FBFGS doesn't support per-parameter options " "(parameter groups)"
            )
        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self.radius_s = radius_s
        self.radius_y = radius_y
        self.radius_ball = radius_ball
        self.radius_ball_s = radius_ball_s
        self.direction_device = direction_device
        self.optimizer_device = optimizer_device
        self.max_ls= max_ls
        self.max_iter = max_iter
        self.t = 1
        
        # Compute and store offsets for parameters
        self._offsets = []
        offset = 0
        for p in self._params:
            self._offsets.append(offset)
            if torch.is_complex(p):
                offset += 2 * p.numel()
            else:
                offset += p.numel()
        self.prefetch_buffer = prefetch_buffer
        self.norm_group_s = norm_group_s if norm_group_s is not None else len(self._params)  # Default to vector norm (num_layers)
        self.norm_group_y = norm_group_y if norm_group_y is not None else 1  # Default to matrix norm (1)
        self.ro_threshold_rate = ro_threshold_rate
        self.current_ro_threshold = 0  # Start at threshold of 0
        self.y_norms = [] # Initialize y_norms as a direct attribute
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )
        return self._numel_cache
    # Remove _init_flat_params method entirely
    def _split_direction_to_layers(self, flat_tensor):
        """Split flat tensor into chunks corresponding to parameter sizes."""
        param_sizes = [p.numel() for p in self._params]
        chunks = []
        offset = 0
        for size in param_sizes:
            chunks.append(flat_tensor[offset:offset+size])
            offset += size
        return chunks
    def gram_schmidt_orthogonalization(self, flat_grad: Tensor, ball_radius: float = 500) -> Tensor:
        """Decompose gradients into orthogonal and natural opposition components."""
# TODO: detach params
        ball_radius = 50
        flat_grad = flat_grad.to(torch.float64)
        param_chunks = self._split_direction_to_layers(flat_grad)
        
        adjusted_chunks = []
        i = 0
        frozen_layers = 0
        for p, grad_chunk in zip(self._params, param_chunks):
            i += 1
            param = p.detach().view(-1).to(torch.float64)
            grad = grad_chunk.to(torch.float64)
           
            param_sq_norm = torch.dot(param, param)
            proj_coeff = torch.dot(grad, param) / param_sq_norm
            ortho_component = grad - proj_coeff * param
            oppose_component = min(proj_coeff, 0.0) * param
            ortho_mag = torch.sqrt(torch.dot(ortho_component, ortho_component))
            oppose_mag = torch.sqrt(torch.dot(oppose_component, oppose_component))
#            if oppose_mag < ortho_mag and proj_coeff < 0 and torch.sqrt(param_sq_norm) > ball_radius:
# Bleed the freak
            if oppose_mag < ortho_mag and proj_coeff < 0 and torch.sqrt(param_sq_norm) > ball_radius:
              ratio = ortho_mag/oppose_mag
              oppose_component = oppose_component * ratio
#            if proj_coeff >= 0:
#              ortho_component = 0.5*ortho_component
            
              combined = ortho_component + oppose_component
#              combined =  oppose_component
              adjusted_chunks.append(combined)
            else:
              combined = ortho_component + oppose_component
              adjusted_chunks.append(combined)
            
        return torch.cat(adjusted_chunks).to(dtype=flat_grad.dtype).contiguous()
    def _split_direction_to_groups(self, flat_tensor, norm_group=None):
        """Split flat tensor into groups based on norm_group parameter.
        
        norm_group behavior:
        - norm_group == 1: One group per layer (matrix norm, current behavior)
        - norm_group == num_layers: One group for all parameters (vector norm on flat_grad)
        - norm_group < 1: Split each layer into int(1/norm_group) groups (sub-layer norm)
        - norm_group > num_layers: Split parameters into norm_group groups
        - norm_group == 0: Treat as 1 (fallback to matrix norm)
        """
        param_sizes = [p.numel() for p in self._params]
        total_size = sum(param_sizes)
        num_layers = len(self._params)
        
        # Use provided norm_group or fall back to instance variable
        effective_norm_group = norm_group if norm_group is not None else self.norm_group
        
        # Handle norm_group == 0 as norm_group == 1
        if effective_norm_group == 0:
            norm_group = 1
        else:
            norm_group = effective_norm_group
            
        # Determine number of groups
        if norm_group == 1:
            # Matrix norm: one group per layer (current behavior)
            groups = []
            offset = 0
            for size in param_sizes:
                groups.append(flat_tensor[offset:offset+size])
                offset += size
            return groups
        elif norm_group == num_layers:
            # Vector norm on flat_grad: one group for all parameters
            return [flat_tensor]
        elif norm_group < 1:
            # Split each layer into int(1/norm_group) groups
            groups_per_layer = int(1 / norm_group)
            if groups_per_layer <= 0:
                groups_per_layer = 1  # Fallback
            groups = []
            offset = 0
            for layer_idx, size in enumerate(param_sizes):
                layer_tensor = flat_tensor[offset:offset+size]
                if groups_per_layer == 1 or size == 0:
                    groups.append(layer_tensor)
                else:
                    # Split this layer into groups_per_layer groups
                    # Use split with sizes to handle uneven splits
                    min_size = size // groups_per_layer
                    remainder = size % groups_per_layer
                    split_sizes = [min_size] * groups_per_layer
                    for i in range(remainder):
                        split_sizes[i] += 1
                    
                    layer_groups = torch.split(layer_tensor, split_sizes, dim=0)
                    groups.extend(layer_groups)
                offset += size
            return groups
        else:
#TODO: I think this should just be an error.
            # norm_group > num_layers: Split total parameters into norm_group groups
            num_groups = int(norm_group)
            if num_groups <= 0:
                num_groups = 1  # Fallback
            groups = []
            offset = 0
            group_size = total_size // num_groups
            remainder = total_size % num_groups
            
#TODO: this is wrong the remainder should go with the last layer as a jumbo layer
            for i in range(num_groups):
                # Distribute remainder among first few groups
                current_size = group_size + (1 if i < remainder else 0)
                if current_size > 0:
                    groups.append(flat_tensor[offset:offset+current_size])
                    offset += current_size
            return groups
    def norm_select(self, tensor: Tensor, 
                   norm: int = 2,
                   radius_scaling: float = 1.0,
                   radius_ball: float = 1.0,
                   eps: float = 1e-8,
                   norm_group: Optional[Union[int, float]] = None) -> Tensor:
        """
        Encapsulates the normalization logic based on norm_group parameter.
        Splits the tensor into groups based on norm_group and applies normalization.
        
        norm_group behavior:
        - norm_group == 1: Matrix norm (one group per layer, current behavior)
        - norm_group == num_layers: Vector norm on flat_grad (one group for all parameters)
        - norm_group < 1: Sub-layer norm (split each layer into groups)
        - norm_group > num_layers: Group norm (split parameters into norm_group groups)
        """
        with torch.no_grad():
            # Split tensor into groups based on norm_group
            effective_norm_group = norm_group if norm_group is not None else self.norm_group
            chunks = self._split_direction_to_groups(tensor, norm_group=effective_norm_group)
            
            # Vectorized Phase 1: Parameter-wise scaling (optional) - zero out small elements per group
            if radius_scaling > 0:
                # Compute norms for all chunks at once
                norms = torch.stack([
                    torch.linalg.vector_norm(chk, ord=norm)
                    if chk.numel() > 0 else torch.tensor(0., device=tensor.device)
                    for chk in chunks
                ])
                # Broadcast threshold calculation: radius_scaling * eps * norms
                thresholds =   norms.unsqueeze(1) / radius_scaling  # shape: (num_chunks, 1) / radius_scaling
                # Create mask for each chunk by comparing absolute values with threshold
                masks = [
                    torch.abs(chk) >= threshold.expand_as(chk)
                    if chk.numel() > 0 else torch.zeros_like(chk)
                    for chk, threshold in zip(chunks, thresholds)
                ]
                # Apply masks
                chunks = [chk * mask for chk, mask in zip(chunks, masks)]
            
#TODO: extract this
            # Vectorized Phase 2: Ball projection (optional) - only apply if l2_norm/radius_ball >= 1 so we never increase the vector
            if radius_ball > 0:
                l2_norms = torch.stack([
                    torch.linalg.vector_norm(chk, ord=2)
                    if chk.numel() > 0 else torch.tensor(0., device=tensor.device)
                    for chk in chunks
                ])
                # Only scale down when l2_norm/radius_ball >= 1 (i.e., l2_norm >= radius_ball) to avoid increasing the vector
                # When l2_norm < radius_ball, use factor of 1 (no change)
                factors = torch.where(
                    (l2_norms > eps) & (l2_norms/radius_ball >= 1),
#                    (l2_norms > eps) ,
                    l2_norms / radius_ball,
                    torch.tensor(1.0, device=tensor.device)
                ).unsqueeze(1)  # shape: (num_chunks, 1)
                
                chunks = [
                    chk / factor.expand_as(chk)
                    if chk.numel() > 0 else chk
                    for chk, factor in zip(chunks, factors)
                ]
            
            if len(chunks) == 0:
                return tensor  # Return original if no chunks
            return torch.cat(chunks)
#TODO: allow this to be distributive, such that we can have multiple nodes generate grads and keep them on their respective devices for all of fbfgs.
    def _gather_flat_grad(self):
        views = []
        self._last_penalty = torch.zeros(1).to(self.optimizer_device)
        
        dot_reg = 0
        for p in self._params:
            if p.grad is not None and not p.grad.is_sparse:
                p_flat = p.view(-1).to(p.grad.dtype)
                g_flat = p.grad.view(-1)
                
#                if p_flat.numel() > 0 and g_flat.numel() > 0 :
#                    dot = torch.dot(p_flat, g_flat)
                
                    # Regularize the gradient for the event horizon reduction (negative alignment)
#                    p_mag_sq =  (torch.dot(p_flat, p_flat))
#                    p_mag =  torch.sqrt(torch.dot(p_flat, p_flat))
#                    p_norm =  torch.sqrt(torch.dot(p_flat, p_flat))/50#/ 50
## TODO: we can just subtract the positive projection here? essentially remove it completely and adjust the loss by its magnitude?
#                    if dot > 0 and p_norm > 1:
## TODO: we want this to be adjusted by the param magnitude somehow but need to review and whiteboard this
#                        projection_reg= ((dot/p_mag_sq)* p_flat) *p_mag
##                        projection_reg= torch.dot(projection_reg, projection_reg)
#                        dot_reg += torch.sqrt(torch.dot(projection_reg, projection_reg))#*p_mag
## TODO: ensure the loss is correct. We have the square here for the grad mag
#                        self._last_penalty= self._last_penalty + torch.sqrt(torch.dot(projection_reg, projection_reg))#*p_mag
#                        print("grad mag before: " + str(torch.dot(g_flat, g_flat)))
## TODO: this doesnt preserve the direction angle? need to travel along the vector
#                        p.grad.view(-1).sub_(projection_reg )# TODO: projection_reg */ p_flat?
#                        print("grad mag after: " + str(torch.dot(g_flat, g_flat)))
##                    if dot == 0 and p_norm > 1:# TODO: negative orthogonality is retarded here whereas it would be boosted by gso.
### TODO: this is more aggresive than the positive projection? balance/tune this.
###                        ortho_limiter = torch.dot(g_flat, g_flat) * p_norm
##                        ortho_limiter = g_flat * p_norm*p_mag
##                        self._last_penalty= self._last_penalty + torch.sqrt(torch.dot(ortho_limiter, ortho_limiter))#*p_mag
##                        dot_reg += torch.sqrt(torch.dot(ortho_limiter, ortho_limiter))#*p_mag
##                        p.grad.view(-1).sub_(ortho_limiter)
#                    
            
            view = p.grad.view(-1)
                
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view.to(self.optimizer_device))
            
        grad = torch.cat(views, 0)
        return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    def gather_norm_flat_grad(self, norm=2, radius_ball=1.0):
        """Gather flat gradients and normalize based on norm_group parameter."""
        flat_grad = self._gather_flat_grad()
        # Use norm_select to handle normalization based on norm_group
        normed_grad = self.norm_select(flat_grad, norm=norm, radius_scaling=radius_ball, radius_ball=2.)
        return normed_grad
    def normalize_per_parameter_chunks(self, tensor: Tensor, 
                                     norm: int = 2,
                                     radius_scaling: float = 1.0,
                                     radius_ball: float = 1.0,
                                     eps: float = 1e-8) -> Tensor:
        """
        Vectorized normalization with parallel computations and optional scaling
        Based on norm_group parameter:
        - norm_group == 1: Matrix norm (one group per layer, current behavior)
        - norm_group == num_layers: Vector norm on flat_grad (one group for all parameters)
        - norm_group < 1: Sub-layer norm (split each layer into groups)
        - norm_group > num_layers: Group norm (split parameters into norm_group groups)
        """
        return self.norm_select(tensor, norm=norm, radius_scaling=radius_scaling, radius_ball=radius_ball, eps=eps)
    def _add_grad(self, step_size, update):
        """Perform parameter update with a dense or sparse tensor update."""
        if torch.is_tensor(step_size):
            step_size = step_size.item()
        
        # SparseFlatTensor handling (new logic)
        if isinstance(update, SparseFlatTensor):
            flat_param_copy = torch.nn.utils.parameters_to_vector(self._params)
            SparseFlatTensor._add_sparse_dense_alpha(update, flat_param_copy, alpha=step_size)
            torch.nn.utils.vector_to_parameters(flat_param_copy, self._params)
        
        # Dense tensor handling (original logic)
        else:
            device = torch.device(self.optimizer_device)
            if update.device != device:
                update = update.to(device)
            offset = 0
            for p in self._params:
                numel = p.numel()
                param_update = update[offset:offset+numel]
                p_view = p.view(-1)
                p_view.add_(param_update, alpha=step_size)
                offset += numel
        
        # NaN guard - apply to each parameter
        for p in self._params:
            p.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        return self._numel()
    def _directional_evaluate(self, closure, t, d, saved_params):
        """Evaluate with gradient regularization via second backward pass"""
#        for p, p_saved in zip(self._params, saved_params, strict=True):
#            p.copy_(p_saved)
#        if isinstance(d, SparseFlatTensor):
#            if d.values.device != self.optimizer_device:
#                d = d.to(self.optimizer_device)
#            self._add_grad(t, d)
#        else:
#            self._add_grad(t, d.to(self.optimizer_device))
        self._add_grad(t, d)
            
        loss = closure()
        flat_grad = self._gather_flat_grad()
        
#        print("component losses   Loss: " + str(loss) + " Loss_Reg: " + str(self._last_penalty) +" alpha * Loss_Reg " + str(self.lambda_reg * self._last_penalty))
        
#        loss =  loss + self._last_penalty.to(self.optimizer_device)
        
        for p, p_saved in zip(self._params, saved_params, strict=True):
            p.copy_(p_saved)
            
        return loss, flat_grad
    def sparse_direction_approximate(self, old_stps: list[SparseFlatTensor], old_dirs: list[SparseFlatTensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, y_norms: list[Tensor], optimizer_device: str, t: float, radius_s: float, radius_ball_s: float, norm: float, y_norm: float, ls_failed: bool, orthogonality: float, n_iter: int, norm_group: Optional[Union[int, float]] = None, ro_threshold_val: float = 0) -> tuple[Tensor, Tensor, list[float]]:
        PREFETCH_THRESHOLD_VALUES = self.prefetch_buffer  # Use hyperparameter
        compute_stream = torch.cuda.current_stream()
        transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        num_old = len(old_dirs)
        q = flat_grad.to(torch.float32).to(optimizer_device).neg()
#put on the local ball without selection
        #put on the local ball without selection
        q = self.norm_select(q, norm=2., radius_scaling=0., radius_ball=self.radius_ball, norm_group=self.norm_group_y)
        
        # Normalize each parameter's chunk
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        print("q max value after layer norm: " + str(q.max()))
        q_current_l2_norm = torch.linalg.vector_norm(q, ord=2).item()
        q_inv_norm = 1.0 / (q_current_l2_norm )
        al = torch.empty(num_old, dtype=q.dtype, device=optimizer_device)
        direction_alignment_mask = torch.empty(num_old, dtype=torch.bool, device=optimizer_device)
        direction_similarities = []
        if num_old > 0:
            # Create filtered list of indices where ro[i] >= ro_threshold_val
            valid_indices = []
            for idx in range(num_old):
                if True:#TODO: remove ro_threshold from the code in favor of new ro rewind algorithm
                    valid_indices.append(idx)
            
            # Backward loop with dynamic prefetching over filtered indices
            backward_buffer_dict = {}
            cumulative_values = 0
            filtered_idx = len(valid_indices) - 1  # Start from last valid index
            
            # Initial prefetch until threshold
            while filtered_idx >= 0 and cumulative_values < PREFETCH_THRESHOLD_VALUES:
                i = valid_indices[filtered_idx]
                if i not in backward_buffer_dict:
                    stp = old_stps[i]
                    dir = old_dirs[i]
                    num_values = stp.values.numel() + dir.values.numel()
                    
                    if cumulative_values + num_values > PREFETCH_THRESHOLD_VALUES:
                        break
                    
                    if transfer_stream:
                        with torch.cuda.stream(transfer_stream):
                            stp_device = stp.to(torch.device(optimizer_device), non_blocking=True)
                            dir_device = dir.to(torch.device(optimizer_device), non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(transfer_stream)
                            backward_buffer_dict[i] = (stp_device, dir_device, event, num_values)
                    else:
                        stp_device = stp.to(torch.device(optimizer_device))
                        dir_device = dir.to(torch.device(optimizer_device))
                        backward_buffer_dict[i] = (stp_device, dir_device, None, num_values)
                    
                    cumulative_values += num_values
                filtered_idx -= 1
            
            # Process valid indices in reverse order
            for idx_pos in range(len(valid_indices)-1, -1, -1):
                i = valid_indices[idx_pos]
                start_time = time.time()
                
                # Get current from buffer or prefetch
                if i not in backward_buffer_dict:
                    stp = old_stps[i]
                    dir = old_dirs[i]
                    num_values = stp.values.numel() + dir.values.numel()
                    
                    if transfer_stream:
                        with torch.cuda.stream(transfer_stream):
                            stp_device = stp.to(torch.device(optimizer_device), non_blocking=True)
                            dir_device = dir.to(torch.device(optimizer_device), non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(transfer_stream)
                            backward_buffer_dict[i] = (stp_device, dir_device, event, num_values)
                    else:
                        stp_device = stp.to(torch.device(optimizer_device))
                        dir_device = dir.to(torch.device(optimizer_device))
                        backward_buffer_dict[i] = (stp_device, dir_device, None, num_values)
                    
                    cumulative_values += num_values
                
                stp_device, dir_device, event, num_values = backward_buffer_dict[i]
                if event:
                    wait_start = time.time()
                    event.wait()
                    wait_end = time.time()
                
#EXTRACT ME GEMINI
                q, direction_alignment_mask, direction_similarities, q_inv_norm = _apply_backward_loop_update(
                    q, dir_device, stp_device, self.y_norms, i, orthogonality, al, direction_alignment_mask, direction_similarities, optimizer_device, ro[i], q_inv_norm, self._offsets, self.radius_ball, self.norm_group_y
                )
#END OF EXTRACT ME GEMINI q
#                print("dir align: " + str(direction_alignment_mask))
                torch.cuda.empty_cache()
                
                # Cleanup and prefetch next
                del backward_buffer_dict[i]
                cumulative_values -= num_values
                
                # Prefetch next valid entries until threshold
                next_filtered_idx = idx_pos - 1
                while next_filtered_idx >= 0 and cumulative_values < PREFETCH_THRESHOLD_VALUES:
                    next_i = valid_indices[next_filtered_idx]
                    if next_i not in backward_buffer_dict:
                        stp_next = old_stps[next_i]
                        dir_next = old_dirs[next_i]
                        num_values_next = stp_next.values.numel() + dir_next.values.numel()
                            
                        if cumulative_values + num_values_next > PREFETCH_THRESHOLD_VALUES:
                            break
                            
                        if transfer_stream:
                            with torch.cuda.stream(transfer_stream):
                                stp_next_device = stp_next.to(torch.device(optimizer_device), non_blocking=True)
                                dir_next_device = dir_next.to(torch.device(optimizer_device), non_blocking=True)
                                event_next = torch.cuda.Event()
                                event_next.record(transfer_stream)
                                backward_buffer_dict[next_i] = (stp_next_device, dir_next_device, event_next, num_values_next)
                        else:
                            stp_next_device = stp_next.to(torch.device(optimizer_device))
                            dir_next_device = dir_next.to(torch.device(optimizer_device))
                            backward_buffer_dict[next_i] = (stp_next_device, dir_next_device, None, num_values_next)
                            
                        cumulative_values += num_values_next
                    next_filtered_idx -= 1
                end_time = time.time()
                symbol = "|" if direction_alignment_mask[i].item() else "_"
                print(f"{symbol}", end='', flush=True)
        print("Q max after first loop: " + str(q.max()))
        # q_for_orthogonalization is no longer needed since we're not doing orthogonalization
        # q_for_orthogonalization = q.clone()  # COMMENTED OUT
        d = q.mul(H_diag.to(torch.float32))
        del q
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if num_old > 0:
            aligned_indices = torch.nonzero(direction_alignment_mask).squeeze(1)
            if aligned_indices.numel() > 0:
                # Forward loop with dynamic prefetching
                forward_buffer_dict = {}
                cumulative_values_fwd = 0
                aligned_indices_list = aligned_indices.cpu().tolist()
                
                # Prefetch aligned indices
                for idx in aligned_indices_list:
                    if idx in forward_buffer_dict:
                        continue
                    stp = old_stps[idx]
                    dir = old_dirs[idx]
                    num_values = stp.values.numel() + dir.values.numel()
                    
                    if cumulative_values_fwd + num_values > PREFETCH_THRESHOLD_VALUES:
                        break
                    
                    if transfer_stream:
                        with torch.cuda.stream(transfer_stream):
                            stp_device = stp.to(torch.device(optimizer_device), non_blocking=True)
                            dir_device = dir.to(torch.device(optimizer_device), non_blocking=True)
                            event = torch.cuda.Event()
                            event.record(transfer_stream)
                            forward_buffer_dict[idx] = (stp_device, dir_device, event, num_values)
                    else:
                        stp_device = stp.to(torch.device(optimizer_device))
                        dir_device = dir.to(torch.device(optimizer_device))
                        forward_buffer_dict[idx] = (stp_device, dir_device, None, num_values)
                    
                    cumulative_values_fwd += num_values
                for k in range(len(aligned_indices_list)):
                    idx = aligned_indices_list[k]
                    if idx not in forward_buffer_dict:
                        stp = old_stps[idx]
                        dir = old_dirs[idx]
                        num_values = stp.values.numel() + dir.values.numel()
                        
                        if transfer_stream:
                            with torch.cuda.stream(transfer_stream):
                                stp_device = stp.to(torch.device(optimizer_device), non_blocking=True)
                                dir_device = dir.to(torch.device(optimizer_device), non_blocking=True)
                                event = torch.cuda.Event()
                                event.record(transfer_stream)
                                forward_buffer_dict[idx] = (stp_device, dir_device, event, num_values)
                        else:
                            stp_device = stp.to(torch.device(optimizer_device))
                            dir_device = dir.to(torch.device(optimizer_device))
                            forward_buffer_dict[idx] = (stp_device, dir_device, None, num_values)
                        
                        cumulative_values_fwd += num_values
                    
                    stp_device, dir_device, event, num_values = forward_buffer_dict[idx]
                    if event:
                        wait_start = time.time()
                        event.wait()
                        wait_end = time.time()
                    
                    d = _apply_forward_loop_update(d, stp_device, dir_device, al, idx, ro[idx], self._offsets, self.radius_ball_s, self.norm_group_s)
#                    d = _apply_forward_loop_update(d, stp_device, dir_device, al, idx, ro[idx], self._offsets, self.radius_ball, self.norm_group_y)
                    
                    # Cleanup
                    del forward_buffer_dict[idx]
                    cumulative_values_fwd -= num_values
                    
                    # Prefetch next aligned indices
                    next_k = k + 1
                    while next_k < len(aligned_indices_list) and cumulative_values_fwd < PREFETCH_THRESHOLD_VALUES:
                        next_idx = aligned_indices_list[next_k]
                        if next_idx in forward_buffer_dict:
                            next_k += 1
                            continue
                        stp_next = old_stps[next_idx]
                        dir_next = old_dirs[next_idx]
                        num_values_next = stp_next.values.numel() + dir_next.values.numel()
                        
                        if cumulative_values_fwd + num_values_next > PREFETCH_THRESHOLD_VALUES:
                            break
                        
                        if transfer_stream:
                            with torch.cuda.stream(transfer_stream):
                                stp_next_device = stp_next.to(torch.device(optimizer_device), non_blocking=True)
                                dir_next_device = dir_next.to(torch.device(optimizer_device), non_blocking=True)
                                event_next = torch.cuda.Event()
                                event_next.record(transfer_stream)
                                forward_buffer_dict[next_idx] = (stp_next_device, dir_next_device, event_next, num_values_next)
                        else:
                            stp_next_device = stp_next.to(torch.device(optimizer_device))
                            dir_next_device = dir_next.to(torch.device(optimizer_device))
                            forward_buffer_dict[next_idx] = (stp_next_device, dir_next_device, None, num_values_next)
                        
                        cumulative_values_fwd += num_values_next
                        next_k += 1
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        effective_norm_group = norm_group if norm_group is not None else self.norm_group_s
#        d = self.norm_select(d, norm=norm, radius_scaling=radius_s, radius_ball=self.radius_ball_s, norm_group=effective_norm_group)
#        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    # gram_schmidt_orthogonalization(self, flat_grad: Tensor, l2_threshold: float = 250.0) -> Tensor:
#        d = self.gram_schmidt_orthogonalization(d)
        d = self.norm_select(d, norm=norm, radius_scaling=radius_s, radius_ball=self.radius_ball_s, norm_group=effective_norm_group)
#        d = self.gram_schmidt_orthogonalization(d)
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        d = d.to(torch.float16)
        return d, direction_alignment_mask, direction_similarities
    def dense_direction_approximate(old_stps: list[Tensor], old_dirs: list[Tensor], ro: list[Tensor], flat_grad: Tensor, H_diag: Tensor, optimizer_device: str, t: float, radius_alpha: float, norm: float) -> Tensor:
        torch.cuda.synchronize() # Ensure all previous CUDA operations are complete, especially non-blocking transfers to calculation device
        num_old = len(old_dirs)
        hit_miss = str("")
        similarity = 0.
#TODO: underflow also this should be better formulated and we should try to avoid another hyperparam but arbitrary literals are worse than hyperparams
        if t < 1:
          similarity = similarity/t
        q = flat_grad.neg().to(flat_grad.device)
        # Use norm_select for normalization instead of manual division
        q = self.norm_select(q, norm=2, radius_scaling=1.0, radius_ball=2.)
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
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
        print(hit_miss)
        # Use norm_select for normalization instead of manual division
        d = self.norm_select(d, norm=norm, radius_scaling=radius_alpha, radius_ball=self.radius_ball_s)
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
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
      max_iter = group["max_iter"]
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
      ro_thresholding = 1
      self.saved_params = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
      # NOTE: FBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      state = self.state[self._params[0]]
      # evaluate initial f(x) and df/dx
      # For first iteration, get regularized gradient via _directional_evaluate
      orig_loss = closure()
      # Add regularization to the loss since _gather_flat_grad applies it to gradients
      flat_grad = self._gather_flat_grad()
      loss = orig_loss+ self._last_penalty
      # Use already computed penalty (already tracked grad modifications)
#      loss = torch.tensor(loss ,
#                                      device=self.optimizer_device, 
#                                      requires_grad=True)
#      
#      # Single backward(retain_graph=True, create_graph=True)ass through the loss (accumulate gradients)
#      loss.backward()
#TODO: does this need to be cloned? also in direction evaluate
#      flat_grad = self._gather_flat_grad().clone()  # Capture gradients before zeroing
#      flat_grad = self._gather_flat_grad()
# TODO: wat
      loss = orig_loss+ self._last_penalty
      orig_loss = orig_loss+ self._last_penalty
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
#      flat_grad = state.get("flat_grad", None)
      if "H_diag" in state:
        H_diag = state.get("H_diag")
      else:
        H_diag = 1
        H_diag = torch.tensor(H_diag)
      H_diag = 1
      H_diag = torch.tensor(H_diag)
      self.t = 1
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
      # Restore any entries from recycle_bin
      if "recycle_bin" in state:
          recycle_bin = state["recycle_bin"]
          while recycle_bin:
              entry = recycle_bin.pop()
              idx = entry['index']
              old_dirs.insert(idx, entry['dir'])
              old_stps.insert(idx, entry['stp'])
              ro.insert(idx, entry['ro'])
#              if 'y_norm' in entry:
#                  self.y_norms.insert(idx, entry['y_norm'])
      
      any_line_search_failed = False  # Track if any line search failed in this iteration
      while n_iter < max_iter: # Enforce max_iter
#          saved_params = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
#          if ro and len(ro) > 0:
#              # Use current_ro_threshold instead of ro_threshold_rate
#              print(f"self.current_ro_threshold values count: {self.current_ro_threshold}")
#              if ro:
#                  if self.current_ro_threshold > 0:
#                      k = self.current_ro_threshold
#                      actual_k = min(k, len(ro))
#                      if actual_k > 0:
#                          ro_values = torch.stack(ro)
#                          # Sort ro values in descending order (largest first) and get sorted indices
#                          sorted_ro, sorted_indices = torch.sort(ro_values, descending=True, dim=0)
#                          # Get the nth largest value (0 = largest, 1 = second largest, etc.)
#                          # Since sorted_ro is in descending order, the nth largest is at index actual_k - 1
#                          nth_index = max(0, int(actual_k) - 1)  # Ensure we don't go out of bounds and index is integer
#                          ro_threshold_val = sorted_ro[nth_index].item()  # Convert to Python scalar
##                          print(f"Sorted ro values: {sorted_ro}")
#                          print(f"ro_threshold_val: {ro_threshold_val}")
#                          print(f"ro values max: {sorted_ro[0].item()}, min: {sorted_ro[-1].item()}")
##                          print(f"Selected index: {nth_index}, original index: {sorted_indices[nth_index].item()}")
#                      else:
#                          ro_threshold_val = 0
#                  else:
#                      ro_threshold_val = 0
#              else:
#                  ro_threshold_val = 0
#          else:
          ro_threshold_val = 0
          torch.cuda.empty_cache() # Clear cache before direction calculation
          # keep track of nb of iterations
          n_iter += 1
          stored_dirs = len(old_dirs)
          print(f"iteration: {n_iter} (stored directions: {stored_dirs})")
          print("[CRAM]")
          ############################################################
          # compute gradient descent direction
          ############################################################
          # If this is the first iteration or history was reset
#TODO: add a special condition such that if num iters is 1 we start with the direction otherwise we do the gradient.
          if  n_iter== 1 or prev_flat_grad is None:
#          if prev_flat_grad is None:
              restart = False # Flag for restart
              print("RESET (n_iter=1 or prev_flat_grad is None)")
#TODO: this is wrong since it uses the grad from failed linesearch but it manages to wiggle the gradient out of being unstuck a lot. We should either analyze why this tends to work or remove it but if we remove it we need to return if we are on the gradient descent since it will fail deterministically
#              flat_grad = self._gather_flat_grad().to(self.optimizer_device)
#TODO: clip_grad_norm by the l1 norm for a max norm of 1e9 (if needed)
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9)
              q = flat_grad.neg()
              max_abs_grad = flat_grad.abs().max()
              max_abs_q = q.abs().max()
              if max_abs_grad <= tolerance_change:
                print(f"Exiting: max gradient element {max_abs_grad} < tolerance {tolerance_change} (q max: {max_abs_q})")
                return orig_loss
              H_diag = 1
              H_diag = torch.tensor(H_diag, device=self.optimizer_device) # Ensure H_diag is on optimizer_device
              torch.cuda.empty_cache() # Clear cache before history update
              # Calculate the top k ro threshold if we have history
#TODO: clean this up
              if len(old_dirs) != 0  : # or n_iter != 1 :
                d, direction_alignment_mask, direction_similarities = self.sparse_direction_approximate(
                    old_stps, old_dirs, ro, flat_grad, H_diag, self.y_norms, optimizer_device=self.optimizer_device, 
                    t=t, radius_s=self.radius_s, radius_ball_s=self.radius_ball, norm=norm, 
                    y_norm=y_norm, ls_failed=ls_failed, orthogonality=orthogonality, n_iter=new_ys_x, 
                    norm_group=self.norm_group_s, ro_threshold_val=ro_threshold_val
                )
              else:
                d, direction_alignment_mask, direction_similarities = self.sparse_direction_approximate(
                    [], [], [], flat_grad, H_diag, self.y_norms, optimizer_device=self.optimizer_device, 
                    t=t, radius_s=self.radius_s, radius_ball_s=self.radius_ball, norm=norm, 
                    y_norm=y_norm, ls_failed=ls_failed, orthogonality=orthogonality, 
                    n_iter=n_iter, norm_group=self.norm_group_s, ro_threshold_val=0
                )
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
              # Calculate y_dense using clone and in-place operations to reduce allocations
              # Apply L2 norm clipping to flat_grad and prev_flat_grad
              # prev_flat_grad is moved to optimizer_device for calculation, then immediately deleted
              # to free up memory.
              y_dense = flat_grad.clone().to(torch.float32) # y_dense is on flat_grad's device (optimizer_device)
              y_dense.sub_(prev_flat_grad.to(torch.float32).to(self.optimizer_device)) # Ensure prev_flat_grad.to(torch.float32) is on optimizer_device
              y_dense= y_dense.to(torch.float32)
#TODO: y going to epsilon is really the problem for ys. ys cant be below a certain amount due to the obvious calculations involved but if y is very small, the approximation blows up with false curvature etc. due to precision inaccuracies.
#TODO: consider skipping based on y and setting ys thresholding based strictly on the numeric stability of the calculations. Need to think about this..
              s_sparse = d * t #Define s_sparse here
              ys = SparseFlatTensor.sparse_dot_dense(s_sparse, y_dense)
              if ys > 0:
                y_squared = y_dense.dot(y_dense)
                H_diag = ys / y_squared  
                del y_squared  
              else:
                H_diag = 1.
                H_diag = torch.tensor(H_diag, device=self.optimizer_device)  
              # Powell dampening modification
              original_y_dtype = y_dense.dtype # Store original dtype
              torch.cuda.empty_cache() # Clear cache
              s_mask = s_sparse.get_nonzero_mask()  # Use the new method
              ys_dense = y_dense.clone()
              ys_dense[~s_mask] = 0
              #Shotgun Noise
              # Use norm_select for normalization instead of manual per-parameter normalization
              print("y dense raw before ops: " + str((y_dense != 0).sum()))
              y_dense = self.norm_select(y_dense, norm=y_norm, radius_scaling=self.radius_y, radius_ball=self.radius_ball, norm_group=self.norm_group_y)
              y_mask = (y_dense != 0)
              print("y dense pre s-mask " + str((y_dense != 0).sum()))
              y_dense[s_mask] = 0
              print("y dense post s-mask " + str((y_dense != 0).sum()))
              print("s mask: " + str((s_mask!=0).sum()))
              print("ys:: " + str((ys_dense!=0).sum()))
              y_dense.add_(ys_dense)
              print("y dense + s_mask  " + str((y_dense != 0).sum()))
              # Layerwise normalization for the second step (replacing norm_yf)
              # Calculate global norm for debugging/compatibility
              del ys_dense
              del y_mask
              del s_mask
              torch.cuda.empty_cache()
              gc.collect() # Collect garbage
              print(f"ys: {ys}")
#              # Powell dampening modification
#              delta =1   #existing curvature threshold
#              if 0 < ys < delta:
##                   Calculate ||s||^2 #efficiently from sparse components
#                  ss = (s_sparse.values**2).sum() + (s_sparse.unit_values**2).sum()
#                  if ss > 1e-10:   #avoid division by zero
#                      theta = (delta - ys) / ss
#                      _add_sparse_dense_alpha(s_sparse, y_dense, alpha=theta, offset=0)
#                      ys = SparseFlatTensor.sparse_dot_dense(s_sparse, y_dense)
#                      print(f"\033[94mApplied Powell dampening. New ys: {ys}\033[0m")
#                  else:
#                      print("Skipped Powell dampening due to small ||s||^2")
#              if self.radius_alpha != 0:
              y_dense = torch.nan_to_num(y_dense, nan=0.0, posinf=0.0, neginf=0.0)
              y_norm_l2 = torch.linalg.vector_norm(y_dense, ord=2.)
              y = SparseFlatTensor.dense_to_sparse_flat_tensor(y_dense.to(torch.float16))
#              s = dense_to_sparse_flat_tensor(s_dense.to(torch.float32))
#              s = dense_to_sparse_flat_tensor(s_sparse)
#              else:
#                y = y_dense
#                s = s_dense
              print("d-delta elements: " + str((d.to_dense() != 0).sum()) + " total: " + str(d.to_dense().numel()), end=' ')
#              print("S elements: " + str((s_dense != 0).sum()) + " total: " + str(s_dense.numel()), end=' ') # Print S elements
              print("y-delta elements: " + str((y_dense != 0).sum()) + " total: " + str(y_dense.numel()), end=' ')
#TODO: this is correct, but maybe there is something more elegant. Possibly reduce based on the mean or the l1/l2 distribution with a hyperparameter. This can be modeled as a outlier distribution problem. We want to maximize Rho so only remove what we need to stabilize the direction-- how we quantify this is TODO
#TODO: this is arguably better than similarity. I wonder if recency matters, such as remove the oldest entries of large Rho (but more elegant)
#TODO: maybe we can even have a weighted pop for the sliding window that considers both the recency and magnitude of the Rho entries? This is all observations on something that needs to be fundamentally quantified.
              # Rho Rewind when ys is too small
              if ys < 1:
                # Ensure recycle_bin is initialized before use
                recycle_bin = state.setdefault("recycle_bin", [])
                old_dirs, old_stps, ro = self._rho_rewind(state, old_dirs, old_stps, ro, direction_similarities)
                ls_failed = True
                state["ls_failed"] = True
#                # Calculate product of ro[i] and direction_similarity[i]
#                ro_products = [abs(ro[i].item() * direction_similarities[i]) 
#                               for i in range(len(ro))]
#                
#                # Calculate total history length including recycle bin
#                total_history_len = len(ro) + len(recycle_bin)
#                # Calculate 10% of total history (minimum 1)
#                rewind_amount = max(1, int(0.1 * total_history_len))
#                # Ensure we don't rewind more than available active history
#                rewind_amount = min(rewind_amount, len(ro))
#                
#                if rewind_amount > 0:
#                    # Sort indices by ro*direction_similarity product descending
#                    ro_product_tensor = torch.tensor(ro_products)
#                    sorted_values, sorted_indices = torch.sort(ro_product_tensor, descending=True)
#                    
#                    # Select top rewind_amount largest ro*direction_similarity products
#                    indices_to_remove = sorted_indices[:rewind_amount].tolist()
#                    
#                    # Sort indices in reverse order to safely remove from lists
#                    indices_to_remove.sort(reverse=True)
#                    
#                    for idx in indices_to_remove:
#                        recycle_entry = {
#                            'index': idx,
#                            'dir': old_dirs.pop(idx),
#                            'stp': old_stps.pop(idx),
#                            'ro': ro.pop(idx),
#                        }
#                        if idx < len(self.y_norms):
#                            recycle_entry['y_norm'] = self.y_norms.pop(idx)
#                        recycle_bin.append(recycle_entry)
#                    print(f"Moved {rewind_amount} largest ro*direction_similarity products to recycle_bin (ys threshold)")
#                
                ls_failed = True
                state["ls_failed"] = True
              # Only add to history if ys meets threshold
              if ys >= self.ro_threshold_rate:  # TODO: is this Kosher?
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
#TODO: this is broken we need a non sleep solution. Possibly just set history size since OS is unreliable.
                        while True:
                            cpu_ram_available = psutil.virtual_memory().available / (1024**3)
                            if cpu_ram_available != old_available_ram_before_pop or not old_dirs:
                                break # Memory changed or history is empty, exit polling
                            time.sleep(0.01) # Small sleep to prevent busy-waiting
                  except Exception as e:
                    print(f"CPU RAM check failed: {e}. Falling back to default memory management.")
                print(f"L-BFGS history popped. History size reduced to: {len(old_dirs)}")
                torch.cuda.empty_cache() # Clear cache before history update
                # Store new direction/step and compute its L2 norm
                y_to_store = y.to(self.direction_device, non_blocking=False, pin_memory=True)
                old_dirs.append(y_to_store)
                old_stps.append(s_sparse.to(self.direction_device, non_blocking=False, pin_memory=True))
                ro.append(torch.tensor([(1. / ys)]))
                # Convert dense y to compute norm
# TODO: calculate this in selection before we sparsify
#                y_dense = y.to_dense()
##                y_norm_l2 = torch.linalg.vector_norm(y_dense, ord=float("inf"))
##                y_dense = y_dense/abs(y_dense).max()
#                y_norm_l2 = torch.linalg.vector_norm(y_dense, ord=2.)
#                self.y_norms.append(1/torch.sqrt(torch.sum(y_dense**2)))
                self.y_norms.append(1/y_norm_l2)
                new_ys_x = new_ys_x + 1
              if n_iter > max_iter or loss == 0:
                self.ro_thresholding = max(1.0 - self.ro_threshold_rate, 0.0)
                state["old_stps"] = old_stps
                state["ro"] = ro
                state["old_dirs"] = old_dirs
                break
              q = flat_grad.neg()
              max_abs_grad = flat_grad.abs().max()
              max_abs_q = q.abs().max()
              if max_abs_grad <= tolerance_change:
# TODO: we probably also need gtd check since we dont abs in sw linesearch
                print(f"Exiting: max gradient element {max_abs_grad} < tolerance {tolerance_change} (q max: {max_abs_q})")
                self.ro_thresholding = max(1.0 - self.ro_threshold_rate, 0.0)
                state["old_stps"] = old_stps
                state["ro"] = ro
                state["old_dirs"] = old_dirs
                break
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
#TODO: TEST THIS!
#              flat_grad = self._gather_flat_grad()
#TODO: may need to try this again? the hessian doesn't pertain as much given that the next direction is likely orthogonal to the last.
#TODO: it might make sense to divide by the history size so we keep curvature normalized to prevent explosions in direction approx.
#              H_diag = 1
#              H_diag = torch.tensor(H_diag)
#              torch.nn.utils.clip_grad_norm_(flat_grad, max_norm=1e9) # Clip gradient norm
#              if self.radius_alpha == 0: # Check if radius_alphaping is disabled
#                d = self.dense_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, direction_device=self.direction_device, t=t, radius_alpha=self.radius_alpha, norm=norm)
#              else:
              d, direction_alignment_mask, direction_similarities = self.sparse_direction_approximate(old_stps, old_dirs, ro, flat_grad, H_diag, self.y_norms, optimizer_device=self.optimizer_device, t=t, radius_s=self.radius_s, radius_ball_s=self.radius_ball, norm=norm, y_norm=y_norm, ls_failed=ls_failed, orthogonality=orthogonality, n_iter=new_ys_x, norm_group=self.norm_group_s, ro_threshold_val=ro_threshold_val )
              state["direction_alignment_mask"] = direction_alignment_mask
              # sparse_direction_approximate already applies norm_select
              del H_diag
#TODO: fix this, we just need to write to hist not calculate everything else but we shouldnt check ys for this condition
#TODO: this or the above should be redundant trace and remove redundancy
#          if n_iter >= max_iter or loss == 0:
#            break
          prev_flat_grad = flat_grad.cpu().clone()# TODO: this may be redundant, to should be clone not copy
          prev_loss = loss
          # The direction d is already normalized by sparse_direction_approximate
          ############################################################
          # compute step length
          ############################################################
          # reset initial guess for step size
          # directional derivative
   #TODO: see if we can get bracketing instead to make this faster, e.g. set to 1 so we start t_prev and t at 0,1 this allows for one of the most interesting aspects of L-BFGS: maximum loss reduction with minimal gradient magnitude (CRAM the model information wise) since we would be preferentially bracketing lowest Strong Wolfe points first in terms of step size
          # Unpack d from tuple before using it
          gtd_sparse_product = flat_grad * d.to(self.optimizer_device) # Ensure d is on optimizer_device
          gtd = gtd_sparse_product.sum()  # g * d
          del gtd_sparse_product
          gc.collect()
          torch.cuda.empty_cache()
          d = SparseFlatTensor.dense_to_sparse_flat_tensor(d)
          #          prev_flat_grad = prev_flat_grad.to(self.direction_device) # This move is handled before y calculation
          t = self.t
          # directional derivative is below tolerance
          #          if gtd > -tolerance_change:
          #              break
          # optional line search: user function
          ls_func_evals = 0
          if line_search_fn is not None:
              # Save parameters before line search
#TODO: instead of saving all the params, save the SparseFlatTensor of params masked by indices of d. Write save and restore dense methods for SparseFlatTensor.
#              saved_params = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
              
              # perform line search, using user function
              if line_search_fn != "strong_wolfe":
                  raise RuntimeError("Only 'strong_wolfe' is supported for line search.")
              else:
                  # Define obj_func with saved_params captured
                  def obj_func(t_step, d_direction):
                      return self._directional_evaluate(closure, t_step, d_direction, self.saved_params)
                  loss_before_ls = loss
                  flat_grad_before_ls = flat_grad
                  prev_loss = loss
                  success, loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                      obj_func, self.direction_device, t, d, loss, flat_grad, gtd, c2=c2, c1=c1, bracket_shift=bracket_shift, bracket_shove=bracket_shove, capture_min_step=capture_min_step, capture_max_step=capture_max_step, optimizer_device=self.optimizer_device , max_ls = self.max_ls
                  )
                  # TODO: consider the armijo condition here to prevent bonking at higher orders (initial norm of 1).
              if not success:
                  for p, p_saved in zip(self._params, self.saved_params):
                      p.copy_(p_saved)
#TODO: there is still a param restore bug here.
                  # Reset parameters to the state before line search
                  print("\033[91mLinesearch failure, retrying with adjusted parameters.\033[0m")
                  # Mark failure and reset step size to 1
                  loss = prev_loss
                  t = torch.tensor(1.)
                  self.t = t.item()  # Also reset class-level step size
                  flat_grad = prev_flat_grad.to(self.optimizer_device)
                  prev_flat_grad = None
#Ro Rewind
                  # Perform Rho Rewind on linesearch failure
                  old_dirs, old_stps, ro = self._rho_rewind(state, old_dirs, old_stps, ro, direction_similarities)
                  # Cleanup: always store direction alignment mask in state
                  state["direction_alignment_mask"] = direction_alignment_mask.detach().cpu()
                  if not old_dirs:
                    return orig_loss
                  # Continue to next iteration to retry
                  continue
              else: # Strong Wolfe line search succeeded
                  ls_failed = False
                  state["ls_failed"] = False # Store ls_failed state
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
#                  if isinstance(d, SparseFlatTensor):
#                      offset = 0
#                      for p in self._params:
#                          numel = p.numel()
#                          if torch.is_complex(p):
#                              p_view = torch.view_as_real(p).view(-1)
#                          else:
#                              p_view = p.view(-1)
#                          # Apply the scaled sparse direction to the dense parameter
#                          # using the new function.
#                          # --- Key Change: Pass the current offset ---
#                          _add_sparse_dense_alpha(d, p_view, alpha=t, offset=offset)
#                          offset += numel
#                  else: # d is a dense Tensor
                  self._add_grad(t, d)
                  self.saved_params = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
# TODO: we need a second closure that just does loss not generate gradients..
# TODO: find what is causing this it could just be a bug somewhere in linesearch or the restore routine
#                  if closure() == loss:
## TODO: maybe check closure == loss instead? as long as we have the right gradient this should be correct but if it is due to numerical instabilities the grad difference may poison the hessian
#                      self.saved_params = [p.clone(memory_format=torch.contiguous_format) for p in self._params]
#                  else:
#                      print("LOSS PARITY FAILURE")
#                      exit()
#                      success = False
#                      for p, p_saved in zip(self._params, self.saved_params):
#                          p.copy_(p_saved)
#                      continue
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
      # Update optimizer state with current history
      state["old_dirs"] = old_dirs
      state["d"] = d
      state["old_stps"] = old_stps
      state["ro"] = ro
      state["prev_flat_grad"] = prev_flat_grad
      state["ls_failed"] = ls_failed # Store ls_failed state
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
            "current_ro_threshold": self.current_ro_threshold, # Save current_ro_threshold
            "y_norms": self.y_norms, # Save y_norms
            "recycle_bin": state.get("recycle_bin", []), # Save recycle_bin
        }
        torch.save(history, filename)
    def _rho_rewind(self, state, old_dirs, old_stps, ro, direction_similarities):
        """Perform Rho Rewind by removing history entries with highest ro values."""
        recycle_bin = state.setdefault("recycle_bin", [])
#TODO: this actually is a perfect fit for a PID control system instead of a 1/max_iter rewind each time with the setpoint being 1 or whatever the ys threshold is. coefficients might be static for a given architecture, hyperparameter set and dataset.
        
        # Extract just ro magnitudes
        ro_magnitudes = [abs(r.item()) for r in ro]
        
        # Calculate total history length including recycle bin
        total_history_len = len(ro) + len(recycle_bin)
        # Calculate 10% of total history (minimum 1)
        rewind_amount = max(1, int(1/self.max_iter * total_history_len))
        # Ensure we don't rewind more than available active history
        rewind_amount = min(rewind_amount, len(ro))
        print("rewinding " + str(rewind_amount) + " of history: " + str(total_history_len))
        
        if rewind_amount > 0:
            # Sort indices by ro value descending
            ro_magnitudes_tensor = torch.tensor(ro_magnitudes)
            sorted_values, sorted_indices = torch.sort(ro_magnitudes_tensor, descending=True)
            
            # Select top rewind_amount largest ro values
            indices_to_remove = sorted_indices[:rewind_amount].tolist()
            
            # Sort indices in reverse order to safely remove from lists
            indices_to_remove.sort(reverse=True)
            
            for idx in indices_to_remove:
                recycle_entry = {
                    'index': idx,
                    'dir': old_dirs.pop(idx),
                    'stp': old_stps.pop(idx),
                    'ro': ro.pop(idx),
                }
#                if idx < len(self.y_norms):
#                    recycle_entry['y_norm'] = self.y_norms.pop(idx)
                recycle_bin.append(recycle_entry)
            print(f"Moved {rewind_amount} largest ro entries to recycle_bin")
        
        return old_dirs, old_stps, ro
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
            history = torch.load(filename, map_location=load_map_location, weights_only=False)
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
            y_norms_from_history = history.get("y_norms", [])
            if isinstance(y_norms_from_history, dict): # Ensure it's a list for JIT
                y_norms_from_history = []
            self.y_norms = [self._move_item_to_device(item, device_obj, non_blocking=False) 
                                for item in y_norms_from_history]
            self.current_ro_threshold = history.get("current_ro_threshold", 0) # Load current_ro_threshold
#            state["recycle_bin"] = [self._move_item_to_device(item, device_obj, non_blocking=False)
#                                    for item in history.get("recycle_bin", [])]
            print(f"FBFGS history loaded from {filename}")
        except FileNotFoundError:
            print(f"History file {filename} not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading FBFGS history from {filename}: {e}. Starting from scratch.")
