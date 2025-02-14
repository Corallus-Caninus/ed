from typing import Optional, Union
import torch
from torch import Tensor
import gc
import psutil

from torch.optim.optimizer import Optimizer, ParamsT

#NOTE: we use 1e+/-16 as the max resolution to avoid NaN in accumulation operations this needs to be solved more robustly. 
# we may not need to accumulate due to the momentum nature (the q and d have to vanish which could compensate the step size explosion in the direction calculation) of the direction calculation but need to ensure this doesnt happen before assuming

__all__ = ["LBFGS"]


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
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
#TODO: c2 = 1 - 1/num_iterations #we always solve given c2 reduction each data point the exact number required
#    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
    obj_func, x, t, d, f, g, gtd, c1=1e-5, c2=0.9, tolerance_change=1e-16, max_ls=25, bracket_shift=(1/3), bracket_shove=(1/3), capture_min_step=1., capture_max_step=100
):
#TODO: this irks the mathematician in me.
    if c2 == 0:
      c2 = 0.25
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
#    g = g.clone(memory_format=torch.contiguous_format)#.to("cpu")
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
#TODO: why don't we scale d by t here, especially since we are normalizing?
    gtd_new_sparse_product = g_new.to("cuda") * d.to("cuda")
    gtd_new = gtd_new_sparse_product.sum()
    del gtd_new_sparse_product
#    g_new = g_new#.to("cpu")
#    gtd_new = gtd_new.to("cpu")
    t_orig = t
    success = False

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0

    t_best = t 
    f_best = f
    g_best = g
    best_c1 = 0
    best_c2 = 0
    ls_iter=0
    stall_wolfe=0

    while ls_iter < max_ls:
#TODO: we can calculate the delta here for insta wolfes and adjust t by the difference, essentially measuring the drift of the interpolation to see if its shifting left or right to try to stay in the min as long as possible over time
#TODO: e.g.: if wolfe is increasing shift up t, if armijo is increasing, shift down t. We may be able to formulate this as a liner equation or a ratio
        # check conditions
        if  (f_new > (f + c1 * t * gtd.to("cuda"))) or f_new > f_best : #or (ls_iter > 1 and f_new >= f_prev)) : #NOTE: Ward condition
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
#            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_g = [g_prev, g_new]
            bracket_gtd = [gtd_prev, gtd_new]
            break

#TODO: <= for ward condition should be < and just allow first iteration to not check ward condition
        if abs(gtd_new.to("cuda")) <= -c2 * gtd.to("cuda"): # and f_new <= f_best :
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

        # next step
        t_prev = tmp
        f_prev = f_new
#        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        g_prev = g_new
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new_sparse_product = g_new.to("cuda") * d.to("cuda")
        gtd_new = gtd_new_sparse_product.sum()
        del gtd_new_sparse_product
#        g_new = g_new.to("cpu")
        ls_iter += 1
        #RELAXED WOLFE CONDITION
#        cur_c2 =  abs(gtd_new.to("cuda")) - -gtd.to("cuda")  #TODO: inverted case
#        if cur_c2 <= best_c2 and f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
        if f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
          success = True
          stall_wolfe = 0
#          best_c2 = cur_c2
          t_best = t
          f_best = f_new
          g_best = g_new

    # reached max number of iterations?
    if ls_iter == max_ls:
#TODO: this is actually better, big zoom if we are out of iterations.
#        bracket = [0, t]
#        bracket_f = [f, f_new]
#        bracket_g = [g, g_new]
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
    #NOTE: we wait for bracket to collapse, we dont use max linesearch here, if it takes too long turn the bracket hyperparameters up.
    while not done :
        # line-search bracket is so small
#TODO: extract stall_wolfe hyperparameter
#        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change or ls_iter >= max_ls or stall_wolfe >= 4:   # type: ignore[possibly-undefined]
        if abs(bracket[1] - bracket[0])  < tolerance_change or  stall_wolfe >= 3:   # type: ignore[possibly-undefined]
            print("WOLFE PACK")
            return success, f_best, g_best, t_best, ls_func_evals
            	#TODO: return the wolfe pack here
#            break

        # compute new trial value
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0].to("cuda"),  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1].to("cuda"),
        )
#        bracket_gtd[1].to("cpu"),
#        bracket_gtd[0].to("cpu"),  # type: ignore[possibly-undefined]

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
                    t = t - bracket_shove*(t - displacement)
                    print("punt", end = " ")
                else:
                    displacement = min(bracket) + eps
                    t = t + bracket_shove*(displacement - t)
                    print("punt", end = " ")
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new_sparse_product = g_new.to("cuda") * d.to("cuda")
        gtd_new = gtd_new_sparse_product.sum()
        del gtd_new_sparse_product
#        g_new = g_new.to("cpu")
        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?


        if f_new > (f + c1 * t * gtd.to("cuda")) or f_new >= bracket_f[low_pos] or f_new > f_best: #NOTE: Ward condition
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
#            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_g[high_pos] = g_new  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd.to("cuda") and f_new < f_best: #NOTE: Ward condition #TODO: Ward condition should be < not <=, it should be based on < and if gtd is under a threshold such that we cant get a gtd delta
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
#            cur_c2 =  abs(gtd_new.to("cuda")) - -gtd.to("cuda")  #TODO: inverted case
    #        if cur_c2 < best_c2 && cur_c1 < best_c1:
    #NOTE: relaxed wolfe condition. If we fail to find a wolfe we go for best curvature to condition the Hessian.
#            if cur_c2 <= best_c2 and f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
            if f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
    #          print("---GOT NEW WOLFE PACK---")
    #          best_c1 = cur_c1
              success = True
              stall_wolfe = 0
#              best_c2 = cur_c2
              t_best = t
              f_best = f_new
              g_best = g_new

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
#            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  
            bracket_g[low_pos] = g_new
# type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new
        stall_wolfe += 1
        if stall_wolfe >= 3:
          print("STALL WOLFE")


    # return stuff
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    return success, f_new, g_new, t, ls_func_evals


class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

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
        history_size: int = 100,
        c1: float = 1e-3,
        c2: float = 0.25,
        line_search_fn: Optional[str] = None,
        bracket_shift: float =(1/3),
        bracket_shove: float =(1/3),
        capture_min_step: float =1.,
        capture_max_step: float =100,
        gradient_clop: float = 1e-7,
        direction_clop: float = 5e-7
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
            gradient_clop=gradient_clop,
            direction_clop=direction_clop
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
        self.gradient_clop = gradient_clop
        self.direction_clop = direction_clop
        self.t = 1

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache

    # gather flat grads with L2 Normalization
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
        norm = torch.linalg.vector_norm(grad, 1)
        grad = grad/norm
#        return torch.cat(views, 0).to("cpu")
        return grad #.to("cpu")
#TODO: clip out NaN based on dtype max value
#        return grad_raw #.to("cpu")

    # gather flat grads with L2 Normalization
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
        norm = torch.linalg.vector_norm(views, 1)
#        norm = views.max()
        views = views/norm
#TODO: does l1 need a norm scaling parameter or does it naturally scale since it has to sum to one anyways (values that are essentially 0 dont add anything to the norm so it should automatically balance). We may also want a scaling value since large networks might end up clopping too much or even dropping too much with l1. Can we tune this normal scaling value with the same hyperparameter used for clopping s.t. its a hyperparameter that is proportional to a "sub net size"? Probably cant just be one hyperparameter, but can we pass in values 0>x<1? essetially the l0.5 norm for scaling up a bit to account for precision losses? Test this but likely we need a hyperparameter to scale the norm we got from l1.
#TODO: what if we normaling by the max value and let clopping handle what the l1 would do anyways? we would only need to tune the clopping hyperparameter and would get essentially what we want with l1
        #Clop
#TODO: may be worth taking the top K here to have deterministic memory, do this after clopping to create a floor for allocation since we want to allow very sparse outlier gradients
        if isClop:
#          mask = torch.logical_and(views > -1e-4,views < 1e-4)
#          _, indices = torch.topk(views, k=10000000)
#          mask = torch.zeros_like(views)
#          views[indices] = 0
#          mask[indices] = 1
#          views = views*mask
#          mask = mask.view_as(views)
#          print("GRAD:  filtered elements: " + str( mask.sum()  ))
#          views = views[mask]
#          print("filtered: " + str(views[views!=0]))
          views[torch.logical_and(views > -self.gradient_clop,views < self.gradient_clop)] = 0
          print("gradient elements: " + str((views != 0).sum()) + " total: " + str(views.numel()), end=' ')
#          views[torch.logical_and(views > -1e-8,views < 1e-8)] = 0
          views = views.to_sparse()
#NOTE: layer width can be greater than precision for l1 norm. Look here for vanishing l1 viewsient if it occurs.
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
                view_indices = sparse_indices[:, mask] - offset # Adjust indices to be relative to the view
                view_values = sparse_values[mask]
                view = torch.sparse_coo_tensor(view_indices, view_values, torch.Size([numel]), dtype=update.dtype, device=update.device)

                p_flat = p.view(-1)
                if view_values.numel() > 0:  # Check if there are any values to update
                    index = view_indices[0, :]  # Get the indices for index_add_
                    p_flat.index_add_(0, index, view_values * step_size)  # Use index_add_ for vectorized update


            else: #dense path for non-sparse tensors just in case
                view = update.to("cuda")[offset : offset + numel]
                # view as to avoid deprecated pointwise semantics
                p.add_(view.view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

#TODO: we can just clone the bitmask of the sparse gradients since those are the only params we are going to modify
    def _clone_param(self):
#        return [p.clone(memory_format=torch.contiguous_format).to("cpu") for p in self._params]
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
#        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d.to("cuda"))
        loss = float(closure())
        flat_grad = self._gather_norm_flat_grad(1, True)
#        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

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

      # NOTE: LBFGS has only global state, but we register it as state for
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

      flat_grad = self._gather_norm_flat_grad(1, True)
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

#TODO: put old_dirs, steps and ro on CPU. Perform the direction calculation as efficiently as possible with this constraint so we can use main memory for history size
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
      d = flat_grad.neg()
      t = 1
      # optimize for a max of max_iter iterations
      while n_iter < max_iter:
          # keep track of nb of iterations
          gc.collect()
          n_iter += 1
          print("[CRAM]")

          ############################################################
          # compute gradient descent direction
          ############################################################
          #TODO: DEPRECATED, the reset logic should be extracted, this should just be initializing d as grad etc.
#TODO: or if history is empty. Better if we do this by history in case we reset the approximation.
          if prev_flat_grad is None:
#          if n_iter == 1:
              print("RESET")
#              flat_grad_sparse = self._gather_norm_flat_grad(1, True)
              d = flat_grad.neg()
#              prev_flat_grad  = None
#              old_dirs = []
#              old_stps = []
#              ro = []
#              if "old_dirs" in state:
#                state["old_dirs"].clear()
#                state["old_stps"].clear()
#                state["ro"].clear()
              H_diag = 1
              t = 1
              gc.collect()
          else:
#              flat_grad = self._gather_norm_flat_grad(1, True)
#              flat_grad = self._gather_norm_flat_grad(2, False)
              # do lbfgs update (update memory).to("cpu")
              y = flat_grad.to("cuda").sub(prev_flat_grad.to("cuda"))
              s = (d.to("cuda").mul(t))
              ys_sparse_product = y * s
              ys = ys_sparse_product.sum()#y*s
              del ys_sparse_product
#TODO: SCALE HESSIAN^-1 COMPONENTS BY ERROR TO REFINE APPROXIMATION MORE EFFICIENTLY
#TODO: with normalization, armijo should be able to solve s.t. c1 <= 1 since loss reduction is 1:1 if the direction approx is 100% accurate since direction is normalized. We also can expect flat_grad.dot(d) to be 0 if approx is 100% accurate since we set number of iterations based on c2 condition convergence minima. e.g.: c2 = 0.9 we do 10 iterations for 100% reduction.
		#TODO: ys = flat_grad.dot(d)  * ys ? #TODO: (abs(gtd_prev) - -gtd ) * ys TODO: which  of these is better? they both make sense to me right now
#              if ys > set this to 1e-10: #TODO:  this may not work with normalized unit vector failsafe. 1e-16 or precision of assigned dtype or better yet ys > 0
              if ys > 0.0: 
                # updating memory
#                if len(old_dirs) <= history_size:
                if torch.cuda.is_available():
                  try:
                    cuda_memory_allocated = torch.cuda.memory_allocated(device=torch.device('cuda')) / 1000000000
                    print(f"CUDA memory allocated: {cuda_memory_allocated} GB, history_size: {history_size} GB") # Debug print
                    if cuda_memory_allocated >= history_size:#TODO: history size is the amount of memory available from the device
                        # shift  history by one (limited-memory)
                        print(f"pop from history.. History size: {len(old_dirs)}", end=' ')
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                  except Exception as e:
                    print(f"CUDA memory check failed: {e}.  Falling back to psutil.")
                # store new direction/step
                old_dirs.append(y.to_sparse().to("cuda")) # NOTE: was cpu
                old_stps.append(s.to_sparse().to("cuda")) # NOTE: was cpu
                ro.append((1.0 / ys).to("cuda")) # NOTE: was cpu #TODO: can we include information on convergence here. This may be an observation of the approximation accuracy. Also consider the alignment (gtd being as close to zero as possible). essentially we would be scaling how much the approximation is influenced by an entry based on its ability to converge.
              # update scale of initial Hessian approximation
#TODO: was this also shifted? check the original implementation
              y_squared_sparse_product = y * y
              y_squared = y_squared_sparse_product.sum()
              del y_squared_sparse_product
              H_diag = ys / y_squared  # (y*y)
              del y_squared


              y = y.to("cuda") #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              s = s.to("cuda") #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              ys = ys.to("cuda") #NOTE: was cpu #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.

              # compute the approximate (L-BFGS) inverse Hessian
              # multiplied by the gradient
              num_old = len(old_dirs)

#              if "al" not in state:
#                state["al"] = [None] * history_size
#              al = [None] * history_size
              al = [None] * num_old
#              al = state["al"]

              # iteration in L-BFGS loop collapsed to use just one buffer
              q = flat_grad.to("cuda").neg()
              for i in range(num_old - 1, -1, -1):
                  # Sparse dot product using element-wise multiplication and sum, replaced to_dense().dot()
                  sparse_product_al = old_stps[i].to("cuda") * ((q.to("cuda")) * ro[i].to("cuda"))
                  al[i] = sparse_product_al.sum() # replaced to_dense().dot()

                  q.add_(old_dirs[i].to("cuda"), alpha=-al[i])
                  al[i] = al[i].to("cuda") #NOTE: was cpu 

          # multiply by initial Hessian
              # r/d is the final direction
              d = r = torch.mul(q, H_diag)
              del q # DEL 5: q is no longer needed after direction d is computed
              del H_diag # DEL 6: H_diag is no longer needed
              for i in range(num_old):
                  # Sparse dot product using element-wise multiplication and sum, replaced to_dense().dot()
                  sparse_product_be = old_dirs[i].to("cuda") * r
                  be_i = sparse_product_be.sum() * ro[i].to("cuda") # replaced to_dense().dot()
                  del sparse_product_be # DEL 7: sparse_product_be is no longer needed

                  r.add_(old_stps[i].to("cuda"), alpha=al[i].to("cuda") - be_i)

          if prev_flat_grad is None : #or state["n_iter"] == 1:
#              prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format).to("cuda") #NOTE: was cpu
              prev_flat_grad = flat_grad.to("cuda") #NOTE: was cpu
          else:
#              prev_flat_grad.copy_(flat_grad).to("cuda") #NOTE: was cpu
              prev_flat_grad = flat_grad.to("cuda") #NOTE: was cpu
          prev_loss = loss
          # normalize the Hessian's direction #TODO: try scaling the Hessian approximation instead of the resultant direction. Can also try to normalize y s and ys in theory inv Hessian computation can overflow (or even underflow) with large history sizes
#TODO: should we be iterating each tensor for norm like in flat_grad?
          total_norm = torch.abs(d.coalesce().values()).sum()
    #TODO: models can have more parameters than precision can support for l1 and this. add a param to scale up the norm accordingly or automatically calculate the scaling parameter to guaruntee enough parameters
          d = d/total_norm
#            print("direction init sparsity: " + str(d[d == 0.0].sum()))
#            Clop
          direction_values = d.coalesce().values()
          mask = torch.logical_and(direction_values > -self.direction_clop, direction_values < self.direction_clop) #TODO: extract to sub_variance hyperparameter
          direction_values[mask] = 0
          print("direction elements: " + str((direction_values != 0).sum()) + " total: " + str(d.numel()), end=' ')
          d = d.coalesce()
          d = torch.sparse_coo_tensor(d.indices(), direction_values, d.size())
          del mask # DEL 9: mask is no longer needed
          del direction_values # DEL 10: direction_values is no longer needed
#          print("DIRECTION: first and last tensors:" + str(d[-10:]) + " " + str(d[:10]))

          ############################################################
          # compute step length
          ############################################################
          # reset initial guess for step size
#TODO:  numerator is a momentum like term that balances the search start point based on if the gradient is vanishing
#TODO:   extract this to a hyperparameter for tolerance_momentum
#          if state["n_iter"] == 1:
#            t = min(1., 1. / flat_grad.to("cuda").abs().sum()) #* lr
#  #          avg = avg / torch.tensor(flat_grad.size(1)).to("cuda")
#  #.div(torch.tensor(flat_grad.size()).to("cuda"))
#          else:
#            avg = flat_grad.to("cuda").abs().mean()
#            #TODO: we should also consider if the direction is vanishing, whichk apparantly can happen such that this t doesnt move more than epsilon and we never zoom phase(?)
#            print("got avg: " + str(avg)) 
#            t = min(5e5, 5e-5/ avg)
#            print("got t: " + str(t))
#          else:
#            t = min(1., 1. / flat_grad.to("cuda").abs().sum()) #* lr
#              t = lr

          # directional derivative
  	#TODO: see if we can get bracketing instead to make this faster, e.g. set to 1 so we start t_prev and t at 0,1 this allows for one of the most interesting aspects of L-BFGS: maximum loss reduction with minimal gradient magnitude (CRAM the model information wise) since we would be preferentially bracketing lowest Strong Wolfe points first in terms of step size
#          flat_grad = self._gather_norm_flat_grad(1, True) TODO: is this right?
          gtd_sparse_product = flat_grad.to("cuda") * d.to("cuda")
          gtd = gtd_sparse_product.sum() # g * d
          del gtd_sparse_product
#          if state["n_iter"] != 1:
#          avg = gtd.abs().mean()
#          print("got avg: " + str(avg)) 
##          t = min(1e16, 1/avg)
          t = self.t #TODO: this should be set based on an average of step sizes or something. We can track what the learning rate should be to increase the speed of bracket search without missing points at lower step sizes.
##            t = min(5e5, 5e-5/ avg)
#          print("got t: " + str(t))

#          flat_grad = flat_grad.to("cpu")
#          gtd=gtd.to("cpu")
#          flat_grad = flat_grad.to("cpu")
#          gtd=gtd
#          d = d.to("cpu") 
#          d = d 
#          t = t.to("cpu") 

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
			#TODO: implement gradient clipping here
                      return self._directional_evaluate(closure, x, t, d)

                  success, loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                      obj_func, x_init, t, d, loss, flat_grad, gtd, c2=c2,c1=c1, bracket_shift=bracket_shift, bracket_shove=bracket_shove, capture_min_step=capture_min_step, capture_max_step=capture_max_step
                  )
#                      obj_func, x_init, t, d, loss, flat_grad, gtd, c2=(1-1/max_iter)
              if not success: #TODO: we chase misprinted lines
                t = 1 #Unit vector until we restore curvature
#                flat_grad = None
                print("Linesearch failure, resetting..")
                loss, flat_grad = obj_func(x_init, t, d)
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
              self.t  = t
              self._add_grad(t, d)
              print("got stepsize: " + str(t) + "  and loss: " + str(loss))
#              opt_cond = flat_grad.abs().max() <= tolerance_grad #TODO: check if this is even possible given normalization. Once verified, rename to point break
#              opt_cond = opt_cond or loss <= 0 #TODO: this should be one order of magnitude above the minimum since we start getting convergence problems when we are very close to the min of precision
              opt_cond =  loss <= 0 #TODO: this should be one order of magnitude above the minimum since we start getting convergence problems when we are very close to the min of precision
          else:
              # no line search, simply move with fixed-step
              self._add_grad(t, d)
              if n_iter != max_iter:
                  # re-evaluate function only if not in last iteration
                  # the reason we do this: in a stochastic setting,
                  # no use to re-evaluate that function here
                  with torch.enable_grad():
                      loss = float(closure())
                  flat_grad = self._gather_flat_grad()
                  opt_cond = flat_grad.abs().max() <= tolerance_grad
                  ls_func_evals = 1

          # update func eval
          current_evals += ls_func_evals
#          state["func_evals"] += ls_func_evals

          ############################################################
          # check conditions
          ############################################################
          if n_iter == max_iter:
              break

#          if current_evals >= max_eval:
#              break

          # optimal condition
#TODO: we may not need this, just let it hit epsilon grad or zero grad for number of iteration times?
#TODO: also, dont exit on loss < 1e-5 as above, let that point break (loss <= 0) condition
          if opt_cond:
              print("GRAD CONVERGE")
              break

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
