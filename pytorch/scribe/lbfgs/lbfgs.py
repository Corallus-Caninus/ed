from typing import Optional, Union
import torch
from torch import Tensor

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
    obj_func, x, t, d, f, g, gtd, c1=0.5, c2=0.8, tolerance_change=1e-16, max_ls=25
#    obj_func, x, t, d, f, g, gtd, c1=1e-8, c2=1e-3, tolerance_change=1e-32, max_ls=20
):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    g = g.clone(memory_format=torch.contiguous_format).to("cpu")
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
#TODO: why don't we scale d by t here, especially since we are normalizing?
    gtd_new = g_new.dot(d.to("cuda"))
    g_new = g_new.to("cpu")
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
        # check conditions
        if  (f_new > (f + c1 * t * gtd.to("cuda"))): #or (ls_iter > 1 and f_new >= f_prev)) :
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new.to("cuda")) <= -c2 * gtd.to("cuda") and f_new <= f_best: #NOTE: Ward condition 
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
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break


#TODO: increase 100 and consider tuning 0.1 further
        min_step = t + 0.618 * (t - t_prev)#TODO: this can miss, if t+0.01 breaks both armijo and wolfe condition (the interpolation is steep)
        lower_bracket = min(t_prev, t)
        upper_bracket = max(t_prev, t)
        max_step = upper_bracket * 100
#TODO: insufficient progress for bracket maybe?
  
        # interpolate
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev.to("cuda"), t, f_new, gtd_new.to("cuda"), bounds=(min_step, max_step)
        )

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.to("cuda").dot(d.to("cuda"))
        g_new = g_new.to("cpu")
        ls_iter += 1
        #RELAXED WOLFE CONDITION
        cur_c2 =  abs(gtd_new.to("cuda")) - -gtd.to("cuda")  #TODO: inverted case
        if cur_c2 <= best_c2 and f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
          success = True
          stall_wolfe = 0
          best_c2 = cur_c2
          t_best = t
          f_best = f_new
          g_best = g_new.to("cpu")

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
    while not done :
        # line-search bracket is so small
#TODO: extract stall_wolfe hyperparameter
#        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change or ls_iter >= max_ls or stall_wolfe >= 4:   # type: ignore[possibly-undefined]
        if abs(bracket[1] - bracket[0])  < tolerance_change or  stall_wolfe >= 4:   # type: ignore[possibly-undefined]
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
        eps = 0.1 * (max(bracket) - min(bracket))
#        eps = tolerance_change * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 1/3 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    displacement = max(bracket) - eps
                    t = t - 0.618*(t - displacement)
                    print("punt", end = " ")
                else:
                    displacement = min(bracket) + eps
                    t = t + 0.618*(displacement - t)
                    print("punt", end = " ")
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.to("cuda").dot(d.to("cuda"))
        g_new = g_new.to("cpu")
        ls_iter += 1 #TODO: how can we ensure the bracket length is sufficiently small that this isn't a terrible worst case?


        if f_new > (f + c1 * t * gtd.to("cuda")) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd.to("cuda") and f_new < f_best: #NOTE: Ward condition
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
            cur_c2 =  abs(gtd_new.to("cuda")) - -gtd.to("cuda")  #TODO: inverted case
    #        if cur_c2 < best_c2 && cur_c1 < best_c1:
    #NOTE: relaxed wolfe condition. If we fail to find a wolfe we go for best curvature to condition the Hessian.
            if cur_c2 <= best_c2 and f_new < f_best and done != True: #NOTE: Ward condition: convergence must be justified by loss reduction else its converging on orthogonal error dissimilarity.
    #          print("---GOT NEW WOLFE PACK---")
    #          best_c1 = cur_c1
              success = True
              stall_wolfe = 0
              best_c2 = cur_c2
              t_best = t
              f_best = f_new
              g_best = g_new.to("cpu")

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  
# type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new
        stall_wolfe += 1
        if stall_wolfe >= 4:
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
        line_search_fn: Optional[str] = None,
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
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._numel_cache = None
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
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        grad_raw = torch.cat(views, 0)
#        norm = torch.linalg.vector_norm(grad_raw, 2)
#        grads = grad_raw/norm
#        return torch.cat(views, 0).to("cpu")
#        return grads #.to("cpu")
#TODO: clip out NaN based on dtype max value
        return grad_raw #.to("cpu")

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                p = torch.view_as_real(p)
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update.to("cuda")[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format).to("cpu") for p in self._params]
#        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d.to("cuda"))
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
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

      # NOTE: LBFGS has only global state, but we register it as state for
      # the first param, because this helps with casting in load_state_dict
      state = self.state[self._params[0]]
      state.setdefault("func_evals", 0)
      state.setdefault("n_iter", 0)

      # evaluate initial f(x) and df/dx
      orig_loss = closure()
      loss = float(orig_loss)
      current_evals = 1
      state["func_evals"] += 1

      flat_grad = self._gather_flat_grad()
#TODO: remove this if we remove gradient normalization.
#      opt_cond = flat_grad.abs().max() <= tolerance_grad #TODO: see TODO below. Can this ever happen with normalization? shouldn't.
      opt_cond = flat_grad.abs().max() <= 0 #TODO: see TODO below. Can this ever happen with normalization? shouldn't.

      # optimal condition
      if opt_cond or loss != loss:# NOTE: this is a NaN check via equivalence
          print("GRAD CONVERGED") #TODO: if we throw out the hessian, will the gradient norm be able to fix this? No, the normalization scalar coeficient is clamped @ 1 so we only scale the norm down.
						#TODO: can we flip the c2 condition to force curvature to escape like momentum?or like a cosine schedule of learning rate based on sub-optimal convergence? ideally we just set c2 correctly but this would be much more robust and easier to tune.
#TODO: instead of resetting, or alongside resetting, flip the linesearch to search for > C2 condition as a momentum factor.
          print("RESET")
          d = flat_grad.neg()
          old_dirs = []
          old_stps = []
          ro = []
          H_diag = 1
          return orig_loss

#TODO: put old_dirs, steps and ro on CPU. Perform the direction calculation as efficiently as possible with this constraint so we can use main memory for history size
      # tensors cached in state (for tracing)
      d = state.get("d")
      t = state.get("t")
      old_dirs = state.get("old_dirs")
      old_stps = state.get("old_stps")
      ro = state.get("ro")
      H_diag = state.get("H_diag")
      prev_flat_grad = state.get("prev_flat_grad")
      prev_loss = state.get("prev_loss")

      n_iter = 0
      # optimize for a max of max_iter iterations
      while n_iter < max_iter:
          # keep track of nb of iterations
          n_iter += 1
          state["n_iter"] += 1
          print("[CRAM]")

          ############################################################
          # compute gradient descent direction
          ############################################################
          #TODO: dont reset, only initialize with this.
          if state["n_iter"] == 1:
              print("RESET")
              d = flat_grad.neg()
              old_dirs = []
              old_stps = []
              ro = []
              H_diag = 1
          else:
              # do lbfgs update (update memory).to("cpu")
              y = flat_grad.to("cuda").sub(prev_flat_grad.to("cuda"))
              s = (d.to("cuda").mul(t))
              ys = y.dot(s)#y*s
#TODO: SCALE HESSIAN^-1 COMPONENTS BY ERROR TO REFINE APPROXIMATION MORE EFFICIENTLY
#TODO: with normalization, armijo should be able to solve s.t. c1 <= 1 since loss reduction is 1:1 if the direction approx is 100% accurate since direction is normalized. We also can expect flat_grad.dot(d) to be 0 if approx is 100% accurate since we set number of iterations based on c2 condition convergence minima. e.g.: c2 = 0.9 we do 10 iterations for 100% reduction.
		#TODO: ys = flat_grad.dot(d)  * ys ? #TODO: (abs(gtd_prev) - -gtd ) * ys TODO: which  of these is better? they both make sense to me right now
#              if ys > set this to 1e-10: #TODO:  this may not work with normalized unit vector failsafe. 1e-16 or precision of assigned dtype or better yet ys > 0
              if ys > 0.0: 
                # updating memory
                if len(old_dirs) == history_size:
                    # shift history by one (limited-memory)
                    old_dirs.pop(0)
                    old_stps.pop(0)
                    ro.pop(0)
   
                # store new direction/step
                old_dirs.append(y.to("cpu"))
                old_stps.append(s.to("cpu"))
                ro.append((1.0 / ys).to("cpu")) #TODO: can we include information on convergence here. This may be an observation of the approximation accuracy. Also consider the alignment (gtd being as close to zero as possible). essentially we would be scaling how much the approximation is influenced by an entry based on its ability to converge.
  
              # update scale of initial Hessian approximation
#TODO: was this also shifted? check the original implementation
              H_diag = ys / y.dot(y)  # (y*y)

              y = y.to("cpu") #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              s = s.to("cpu") #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.
              ys = ys.to("cpu") #TODO: these should be GCD here this just slows stuff down unless py/torch does an optimization pass on it.

              # compute the approximate (L-BFGS) inverse Hessian
              # multiplied by the gradient
              num_old = len(old_dirs)

              if "al" not in state:
                  state["al"] = [None] * history_size
              al = state["al"]

              # iteration in L-BFGS loop collapsed to use just one buffer
              q = flat_grad.to("cuda").neg()
              for i in range(num_old - 1, -1, -1):
                  al[i] = (old_stps[i].to("cuda").dot(q.to("cuda")) * ro[i].to("cuda"))
                  q.add_(old_dirs[i].to("cuda"), alpha=-al[i])
                  al[i] = al[i].to("cpu")

              # multiply by initial Hessian
              # r/d is the final direction
              d = r = torch.mul(q, H_diag)
#              if H_diag != 1: #TODO: this should be freed we are wasting time by moving it to ram
#                H_diag = H_diag.to("cpu")
              for i in range(num_old):
                  be_i = old_dirs[i].to("cuda").dot(r) * ro[i].to("cuda")
                  r.add_(old_stps[i].to("cuda"), alpha=al[i].to("cuda") - be_i)

          if prev_flat_grad is None:
              prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format).to("cpu")
          else:
              prev_flat_grad.copy_(flat_grad).to("cpu")
          prev_loss = loss
#          d=torch.norm(d, 1.)
          # normalize the Hessian's direction #TODO: try scaling the Hessian approximation instead of the resultant direction. Can also try to normalize y s and ys
          total_norm = torch.linalg.vector_norm(
                 d,1.
             )
          d = d/total_norm

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
          gtd = flat_grad.to("cuda").dot(d.to("cuda"))  # g * d
#          if state["n_iter"] != 1:
#          avg = gtd.abs().mean()
#          print("got avg: " + str(avg)) 
##          t = min(1e16, 1/avg)
          t = self.t #TODO: this should be set based on an average of step sizes or something. We can track what the learning rate should be to increase the speed of bracket search without missing points at lower step sizes.
##            t = min(5e5, 5e-5/ avg)
#          print("got t: " + str(t))

#          flat_grad = flat_grad.to("cpu")
#          gtd=gtd.to("cpu")
          flat_grad = flat_grad
          gtd=gtd
#          d = d.to("cpu") 
          d = d 
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
                      obj_func, x_init, t, d, loss, flat_grad, gtd
                  )
              if not success: #TODO: we chase misprinted lines
                t = 1 #Unit vector until we restore curvature
                loss, flat_grad = obj_func(x_init, t, d)
              flat_grad = flat_grad.to("cuda")
              self.t  = t
              self._add_grad(t, d)
              print("got stepsize: " + str(t) + "  and loss: " + str(loss))
              opt_cond = flat_grad.abs().max() <= tolerance_grad #TODO: check if this is even possible given normalization. Once verified, rename to point break
              opt_cond = opt_cond or loss <= 0
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
          state["func_evals"] += ls_func_evals

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

      state["d"] = d
      state["t"] = t
      state["old_dirs"] = old_dirs
      state["old_stps"] = old_stps
      state["ro"] = ro
      state["H_diag"] = H_diag
      state["prev_flat_grad"] = prev_flat_grad
      state["prev_loss"] = prev_loss

      return orig_loss
