import torch
import math
import sys
from torch.optim.optimizer import Optimizer, required


class OnlineNaturalGradient:
    r"""This object is used in the `NGD` object which is the actual optimizer.
    It is derived from the OnlineNaturalGradient object in Kaldi's
    src/nnet3/natural-gradient-online.h, and the ideas are explained in
    "Parallel training of DNNs with Natural Gradient and Parameter Averaging"
    by D. Povey, X. Zhang and S. Khudanpur, ICLR Workshop, 2015.  But the
    way it is implemented here in PyTorch is a little different, because, due
    to how the automatic differentiation works, we can't easily get access
    to the matrix multiplication or summation that gave us this particular
    derivative.  So we basically treat all the "other" dimensions of the
    parameter object as if they were the minibatch dimension.
    """

    def __init__(self, params, axis, alpha=4.0,
                 rank=-1, update_period=4,
                 eta=0.1):
        r"""
      Constructor.
    Arguments:
        params:       The parameters we are going to operating on. Used
                      to get the device, dtype and shape; the parameter
                      values do not matter at this point.
        axis:         A value in the range [0, len(param_shape) - 1],
                      saying which axis of the parameters this object
                      operates on.  The dimension of the low-rank matrix
                      that we are learning will be equal to params.shape[axis].
        alpha:        A smoothing constant which determines how much
                      we smooth the Fisher matrix with the identity;
                      the scale on the identity is the average diagonal
                      element of the un-smoothed Fisher matrix multiplied by
                      alpha.  The value alpha=4.0 tends to work well
                      experimentally in a wide range of conditions (when
                      measured in terms of classification performance
                      on validation data, not as an optimiser), although
                      perhaps alpha=4.0 represents a perhaps larger-than-expected
                      amount of smoothing.
        rank:         The rank of the structured matrix that we will be
                      updating.  If a value <0 is passed in, it
                      will be set to the smaller of (dim+1) / 2, or 80,
                      where dim = params.shape[axis].
        update_period: Determines how frequently (on how many minibatches
                      we update the natural-gradient matrix; the default
                      of 4 is reasonable.
        eta           A Python float strictly between 0 and 1, that determines how fast
                      we update the Fisher-matrix factor. 
                      i.e. F_t = \eta * S_t + (1 - \eta) F_{t-1}, where S_t is the emperical
                      Fisher estimated from the current minibatch.
        """
        assert axis >= 0 and axis < len(params.shape)
        self.axis = axis
        self.dim = params.shape[axis]
        assert(self.dim > 0)
        self.device = params.device
        self.dtype = params.dtype
        self.alpha = alpha
        if rank >= 0:
            assert(rank > 0 and rank < self.dim)
        else:
            rank = min((self.dim + 1) // 2, 80)
        self.rank = rank
        assert(update_period > 0)
        self.update_period = update_period
        assert eta > 0 and eta < 1
        self.eta = eta
        self.rank = rank

        # epsilon and delta are two values involved in making sure the Fisher
        # matrix isn't singular and that we don't get NaN's appearing; we don't
        # make them configurable.
        self.epsilon = 1.0e-10
        self.delta = 5.0e-4

        # t is a counter that records how many times self.precondition_directions()
        # has been called.
        self.t = 0
        # We won't initiailize the members
        # self.W_t, self.rho_t, self.d_t
        # until someone uses the class.
        self.debug = False

    r"""
    This string contains documentation of the data members of this class.  These are not
    part of the public interface, but we document them for clarity and to make it easier
    to understand the internals.

      self.dim  An int containing the dimension >= 1 that this object operates on; this
                is the dimension (num-rows/cols) of the Fisher matrix that this class
                does a multiplication by.
      self.axis An int >= 0 that is the axis of the parameters which we'll be operating
                on (our Fisher-matrix factor is a square matrix of dimension
                dim = params.shape[self.axis]).
      self.device  The torch.device object that the parameter and derivs live on;
                we will keep the inverse-Fisher-matrix factor W_t, and many temporary
                quantities used in the algorithm, on this device for speed.
      self.dtype  The torch.dtype representing the type of the parameters and derivs;
                expected to be float or double.
      self.alpha  A Python float that determines how much we smooth the Fisher matrix
                with the identity; the default (4.0) represents very aggressive
                smoothing, such that all the NGD is really doing is slowing us down
                in the really dominant directions, but not really speeding us up
                in the dimensions where there is very little variation.
      self.rank  [only if self.dim > 1] A python int in the  range [1, self.dim - 1]
                which is the rank of the low-rank-plus-identity approximation
                to the Fisher matrix.  If self.dim == 1, this object is a no-op
                and we treat the Fisher matrix as the identity matrix [ 1 ], and
                self.rank (and various other class members) are not defined.
      self.update_period  A Python int that determines how frequently (i.e.
                every how-many minibatches) we update the Fisher-matrix factors.
                The default is 4.
      self.eta  A Python float strictly between 0 and 1, that determines how fast
                we update the Fisher-matrix factor. 
                i.e. F_t = \eta * S_t + (1 - \eta) F_{t-1}, where S_t is the emperical
                Fisher estimated from the current minibatch.
      self.epsilon, self.delta  Python floats equal to 1e-10 and 5e-4, which go into
                determining minimum eigenvalues of the Fisher-matrix approximation,
                mostly to avoid situations where our update would be unstable or
                generate NaN's.  These values aren't user configurable.  See
                the paper.
      self.debug  A python bool; if true, certain debugging code will be activated.

     The following variables change with time:
      self.t    A Python int >= 0, which equals the number of times the function
                self.precondition_directions() has been called by the user.  It
                helps to determine on which iterations we update the Fisher-matrix
                approximation.
      self.W_t  A torch.tensor with shape (self.rank, self.dim), device self.device
                and dtype self.dtype, corresponding to
                a factor of the inverse Fisher matrix; it's W_t in the math, see the paper.
      self.d_t_cpu  A torch.tensor with shape (self.rank), device 'cpu' and
                dtype self.dtype, which contains the eigenvalues of the inverse Fisher
                matrix (or is it the Fisher matrix?)... anyway, it's described in the paper.
      self.rho_t  A Python float that is a scale on the unit part in the Fisher-matrix
               approximation on the current iteration (or maybe its inverse, check the
                paper).
    """

    def precondition_directions(self, deriv):
        r"""
        Implements the main functionality of this class; takes the derivative
        "deriv" and returns the 'preconditioned' derivative.

        This function just reorganizes the dimensions and calls
        _precondition_directions1().
        """
        assert deriv.shape[self.axis] == self.dim
        if self.dim == 1:
            return deriv  # This class is o a no-op in that case.

        # What the following statement does is to switch axes so that the axis
        # we operate on (of dim self.dim) is the last one, call
        # self._precondition_directions1, and then switch the axes back.  All
        # this would be done without changing the underlying data.
        return torch.transpose(self._precondition_directions1(deriv.transpose(-1, self.axis)),
                               -1, self.axis)

    def _precondition_directions1(self, deriv):
        r"""
        Internal version of precondition_directions() that expects
        the axis we operate on to be the last axis in the tensor.  So at this point
        we can proceed as if self.axis == len(deriv.shape) - 1.
        The preconditioned derivative that this function returns
        is in the same format.
        """

        assert deriv.shape[-1] == self.dim

        deriv = deriv.contiguous()
        # The following call combines the leading axes into a single axis,
        # so that _preconditions2 can operate on a tensor of shape
        # (remaining_dims_product, dim), calls _precondition_directions2 and
        # then switches back to the shape this function was called with.
        return self._precondition_directions2(deriv.view(-1, self.dim)).view(deriv.shape)

    def _precondition_directions2(self, deriv):
        r""" This corresponds to PreconditionDirections() in the C++ code,
        except rather than modifying deriv in-place and returning a separate
        scaling factor, it returns the modified deriv with the scaling factor
        already applied.
        """
        # The following assert documents the format requirements on 'deriv'.
        assert (len(deriv.shape) == 2 and deriv.shape[1] == self.dim and
                deriv.dtype == self.dtype and deriv.device == self.device)

        if self.t == 0:
            self._init(deriv)

        initial_product = (deriv * deriv).sum()

        deriv_out = self._precondition_directions3(
            deriv, initial_product)

        final_product = (deriv_out * deriv_out).sum()

        if math.isnan(final_product):
            print("Warning: nan generated in NG computation, returning derivs unchanged",
                  file=sys.stderr)
            # If there are NaNs in our class members now, it would be a problem; in
            # future we might want to add code to detect this an re-initialize,
            # but for now just detect the problem and crash.
            self._self_test()
            return deriv

        # the + 1.0e-30 below is to avoid division by zero if the derivative is zero.
        return deriv_out * torch.sqrt(initial_product / (final_product + 1.0e-30))

    def _precondition_directions3(self, X_t, tr_X_Xt):
        """
        This does the 'really' core part of the natural gradient multiplication and
        (on some frames) it updates our Fisher-matrix estimate.  This corresponds,
        roughly, to PreconditionDirectionsInternal() in the C++ code.


        Arguments:
             X_t:     The matrix of derivatives (X_t in the math), a 2-dimensional
                      PyTorch tensor, expected to be on device self.device, and
                      X_t.shape[1] should equal self.dim.
             tr_X_Xt: The value of trace(X X^T) == (X * X).sum(), as a scalar
                      torch.tensor (i.e., with shape equal to ()).

        Return:
             Returns the matrix of derivatives multiplied by the inverse
             Fisher matrix.
        """
        updating = self._updating()
        self.t = self.t + 1
        rho_t = self.rho_t
        alpha = self.alpha
        eta = self.eta
        rank = self.rank
        dim = self.dim
        W_t = self.W_t
        d_t_cpu = self.d_t_cpu

        H_t = torch.mm(X_t, W_t.transpose(0, 1))  # H_t = X_t W_t^T

        # X_hat_t = X_t - H_t W_t
        X_hat_t = X_t - torch.mm(H_t, W_t)

        if not updating:
            # We're not updating the estimate of the Fisher matrix; we just
            # apply the preconditioning and return.
            return X_hat_t

        J_t = torch.mm(H_t.transpose(0, 1), X_t)  # J_t = H_t^T X_t

        # In the paper, N would be the number of rows in the matrix being
        # multiplied (would be related to the minibatch size).
        # In this version, it would be 1
        #N = 1
        N = X_t.shape[0]
        # we choose the fastest way to compute L_t, which depends
        # on the dimension.
        if N > dim:
            L_t = torch.mm(J_t, W_t.transpose(0, 1))  # L_t = J_t W_t^T
        else:
            L_t = torch.mm(H_t.transpose(0, 1), H_t)  # L_t = H_t^T H_t
        K_t = torch.mm(J_t, J_t.transpose(0, 1))  # K_t = J_t J_t^T

        K_t_cpu = K_t.to('cpu')
        L_t_cpu = L_t.to('cpu')

        # d_t_sum and beta_t are python floats.
        # in the math, d_t_sum corresponds to tr(D_t).
        d_t_sum = d_t_cpu.sum().item()
        beta_t = rho_t * (1.0 + alpha) + alpha * d_t_sum / dim

        # e_t is a torch.tensor with shape (rank,) on the CPU.
        # e_{tii} = 1/(\beta_t/d_{tii} + 1)
        e_t_cpu = 1.0 / (beta_t / d_t_cpu + 1.0)
        sqrt_e_t_cpu = torch.sqrt(e_t_cpu)
        inv_sqrt_e_t_cpu = 1.0 / sqrt_e_t_cpu

        # z_t_scale is an arbitrary value (mathematically it can be anything)
        # which we use to keep Z_t in a reasonable numerical range, since it
        # contains the derivatives to the fourth power which can otherwise get
        # large.  we'll divide by this when creating Z_t, and multiply by it
        # when using Z_t.
        z_t_scale = max(1.0, K_t_cpu.trace().item())
        # see eqn:zt in natural-gradient-online.h (the C++ code).  We are computing,
        # Z_t  = (\eta/N)^2 E_t^{-0.5} K_t E_t^{-0.5}
        #   +(\eta/N)(1-\eta) E_t^{-0.5} L_t E_t^{-0.5} (D_t + \rho_t I)
        #   +(\eta/N)(1-\eta) (D_t + \rho_t I) E_t^{-0.5} L_t E_t^{-0.5}
        #   +(1-\eta)^2 (D_t + \rho_t I)^2                              (eqn:Zt)
        #  [And note that E_t and D_t and I are all diagonal matrices, of which
        #   we store the diagonal elements only].
        # Actually the quantity Z_t_cpu here is equal to Z_t / z_t_scale;
        # the scale helps keep it in a good numerical range.

        d_t_plus_rho_t = d_t_cpu + rho_t
        # Note: torch.ger gives the outer product of vectors.
        inv_sqrt_e_t_outer = ((eta / N)**2 / z_t_scale) * \
            torch.ger(inv_sqrt_e_t_cpu,  inv_sqrt_e_t_cpu)
        outer_product1 = ((eta / N) * (1 - eta) / z_t_scale) * \
            torch.ger(inv_sqrt_e_t_cpu, inv_sqrt_e_t_cpu * d_t_plus_rho_t)

        Z_t_cpu = (K_t_cpu * inv_sqrt_e_t_outer +
                   L_t_cpu * (outer_product1 + outer_product1.transpose(0, 1)) +
                   (((1 - eta)**2 / z_t_scale) * (d_t_plus_rho_t * d_t_plus_rho_t)).diag())

        # do the symmetric eigenvalue decomposition Z_t = U_t C_t U_t^T; we do this
        # on the CPU as this kind of algorithm is normally super slow on GPUs, at least
        # on smallish dimensions (note: rank <= 80, with the default configuration).
        (c, U) = Z_t_cpu.symeig(True)
        # reverse the eigenvalues and their corresponding eigenvectors from largest
        # to smallest.  c_t_cpu still has the scale `1/z_t_scale`.
        c_t_cpu = c.flip(dims=(0,))
        U_t_cpu = U.flip(dims=(1,))

        if self.debug:
            error = torch.mm(U_t_cpu * c_t_cpu.unsqueeze(0),
                             U_t_cpu.transpose(0, 1)) - Z_t_cpu
            assert (error * error).sum() < 0.001 * \
                (Z_t_cpu * Z_t_cpu).sum()
        condition_threshold = 1.0e+06
        c_t_floor = ((rho_t * (1.0 - eta)) ** 2) / z_t_scale
        c_t_cpu = torch.max(c_t_cpu, torch.Tensor(
            [c_t_floor]))  # c_t_cpu.floor_(c_t_floor)
        # sqrt_c_t_cpu no longer has the `1/z_t_scale` factor, i.e. it is now
        # the same value as in the math.
        sqrt_c_t = c_t_cpu.to(self.device).sqrt() * math.sqrt(z_t_scale)
        inv_sqrt_c_t = (1.0 / sqrt_c_t)
        # \rho_{t+1} = 1/(D - R) (\eta/N tr(X_t X_t^T) + (1-\eta)(D \rho_t + tr(D_t)) - tr(C_t^{0.5})).
        rho_t1 = (1.0 / (dim - rank)) * ((eta / N) * tr_X_Xt.item() +
                                         (1.0 - eta) * (dim * rho_t + d_t_sum) -
                                         sqrt_c_t.sum().item())

        floor_val = max(self.epsilon, self.delta * sqrt_c_t.max().item())
        # D_{t+1} = C_t^{0.5} - \rho_{t+1} I,  with diagonal elements floored to floor_val.
        # we store only the diagonal.
        d_t1_cpu = torch.max(sqrt_c_t.to('cpu') - rho_t1,
                             torch.tensor(floor_val, dtype=self.dtype))
        if rho_t1 < floor_val:
            rho_t1 = floor_val

        # Begin the part of the code that in the C++ version was called ComputeWt1.
        # beta_t1 is a python float.
        # \beta_{t+1} = \rho_{t+1} (1+\alpha) + \alpha/D tr(D_{t+1})
        beta_t1 = rho_t1 * (1.0 + alpha) + alpha * d_t1_cpu.sum().item() / dim
        assert beta_t1 > 0
        # Compute E_{t+1} and related quanitities; the formula is the same for
        # E_t above. These are diagonal matrices and we store just the diagonal
        # part.
        e_t1_cpu = 1.0 / (beta_t1 / d_t1_cpu + 1.0)
        sqrt_e_t1 = torch.sqrt(e_t1_cpu.to(self.device))

        w_t_coeff = (((1.0 - eta) / (eta / N)) *
                     (d_t_cpu + rho_t)).to(self.device)
        # B_t = J_t + (1-\eta)/(\eta/N) (D_t + \rho_t I) W_t
        B_t = torch.addcmul(J_t, w_t_coeff.unsqueeze(1), W_t)

        # A_t = (\eta/N) E_{t+1}^{0.5} C_t^{-0.5} U_t^T E_t^{-0.5}
        left_product = torch.tensor(
            eta / N, device=self.device, dtype=self.dtype) * sqrt_e_t1 * inv_sqrt_c_t
        A_t = U_t_cpu.to(self.device).transpose(0, 1) * torch.ger(left_product,
                                                                  inv_sqrt_e_t_cpu.to(self.device))
        # W_{t+1} = A_t B_t
        W_t1 = torch.mm(A_t, B_t)
        # End the part of the code that in the C++ version was called ComputeWt1.

        self.W_t = W_t1
        self.d_t_cpu = d_t1_cpu
        self.rho_t = rho_t1
        # t has been incremented at the top of this function
        if self.debug:
            self._self_test()
        return X_hat_t

    def _self_test(self):
        assert self.rho_t >= self.epsilon
        d_t_max = self.d_t_cpu.max().item()
        d_t_min = self.d_t_cpu.min().item()
        assert d_t_min >= self.epsilon and d_t_min > self.delta * d_t_max * 0.9
        assert self.rho_t > self.delta * d_t_max * 0.9
        beta_t = self.rho_t * (1.0 + self.alpha) + \
            self.alpha * self.d_t_cpu.sum().item() / self.dim

        e_t = (1.0 / (beta_t / self.d_t_cpu + 1.0)).to(self.device)
        sqrt_e_t = torch.sqrt(e_t)
        inv_sqrt_e_t = 1.0 / sqrt_e_t

        should_be_zero = (torch.mm(self.W_t, self.W_t.transpose(0, 1)) *
                          torch.ger(inv_sqrt_e_t, inv_sqrt_e_t) - torch.eye(self.rank, device=self.device))
        assert should_be_zero.abs().max().item() < 0.1

    def _updating(self):
        r""" Returns true if, on this iteration, we are updating the Fisher
        matrix."""
        num_initial_iters = 10
        if self.t < num_initial_iters:
            return True
        else:
            return self.t % self.update_period == 0

    def _init(self, deriv):
        assert self.t == 0
        # _init_default() initializes W_t, rho_t and d_t to values that do
        # not depend on 'deriv'.
        self._init_default()
        # setting self.t to 1 will stop self._precondition_directions2 from
        # recursing to _init.
        self.t = 1
        # the following loop will do the standard update on each the 3
        # iterations, causing the low-rank matrix to get closer to what we would
        # get in a SVD-based initialization...  it's similar to the power
        # method.
        # Each time, we discard the return value of
        # self.precondition_directions2.
        for n in range(0, 3):
            self._precondition_directions2(deriv)

        # The calling code is going to increment self.t again, so reset it
        # to zero which is the value it will have had a entry; then self.t
        # will equal the value
        self.t = 0

    def _init_default(self):
        r"""Called from _init(), this function sets the parameters self.W_t,
        self.rho_t and self.d_t to some default values; they will then be
        updated by several iterations of the standard update but done with
        the same 'deriv'; this is a fast approximation to an SVD-based
        initialization."""
        assert self.rank < self.dim and self.rank > 0 and self.alpha > 0.0
        self.rho_t = self.epsilon
        # d_t will be on the CPU, as we need to do some sequential operations
        # on it.
        self.d_t_cpu = self.epsilon * \
            torch.ones((self.rank,), dtype=self.dtype)
        # W_t is on self.device (possibly a GPU).

        # E_tii is a scalar here, since it's the same for all i.
        E_tii = 1.0 / (2.0 + (self.dim + self.rank) * self.alpha / self.dim)
        self.W_t = math.sqrt(E_tii) * self._create_orthonormal_special()
        assert self.t == 0

    def _create_orthonormal_special(self):
        r"""This function, used in _init_default(), creates and returns a PyTorch
        tensor on device self.device and with dtype self.dtype, with shape
        (self.rank, self.dim) that is like the following:
          [  1.1 0   1   0   1
             0   1.1 0   1   0   ] * k
        where k is chosen so that each row has unit 2-norm.  The motivation is
        that this is faster than starting with a random matrix and
        orthonormalizing it with Gram-Schmidt.  The base matrix it starts with
        the identity times 1.1, then has copies of the identity to fill out the
        remaining dimensions.  The reason for this structure is, to ensure each
        row and column has a nonzero value; the 1.1 is for symmetry breaking
        since there may be architectures where the deriviative in the direction
        [1 1 1 1 .. 1 ] would be zero and having the sum of rows be equal to
        that value might cause the matrix after multiplying by the data derivs
        to be singular, which would put the code on a less efficient path
        involving CPU-based operations."""

        first_elem = 1.1
        num_cols = self.dim // self.rank
        remainder = self.dim % self.rank
        k = torch.ones(self.rank, dtype=self.dtype, device=self.device) * \
            (1.0 / math.sqrt(first_elem * first_elem + num_cols - 1))
        k[:remainder] = torch.ones(
            remainder, dtype=self.dtype, device=self.device) * (1.0 / math.sqrt(first_elem * first_elem + num_cols))
        diag_v = torch.ones(self.rank, dtype=self.dtype,
                            device=self.device) * k
        diag = torch.diag(diag_v)
        first_diag_v = diag_v * first_elem
        first_diag = torch.diag(first_diag_v)
        ans = torch.cat((first_diag, diag.repeat(
            1, num_cols + 1)), 1)[:, :self.dim]

        if self.debug:
            should_be_zero = (torch.mm(ans, ans.transpose(0, 1)) -
                              torch.eye(self.rank, dtype=self.dtype,
                                        device=self.device))
            s = should_be_zero.abs().max().item()
            assert s < 0.1
        return ans


class NGD(Optimizer):
    r"""Implements natural gradient (optionally with momentum).
        In future we may make some of the options of the NG user-modifiable
        but for now we use defaults that have been found to work well.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.NGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    TODO: more information
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 ngd=True, alpha=4, rank=-1, update_period=4, eta=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        ngd=ngd, alpha=alpha, rank=rank,
                        update_period=update_period, eta=eta)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('ngd', True)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            ngd = group['ngd']
            alpha = group['alpha']
            rank = group['rank']
            update_period = group['update_period']
            eta = group['eta']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if ngd:
                    param_state = self.state[p]
                    if 'ngd_dict' not in param_state:
                        ngd_dict = param_state['ngd_dict'] = dict()
                        for axis in range(len(p.shape)):
                            ngd_dict[axis] = OnlineNaturalGradient(
                                p, axis, alpha, rank, update_period, eta)

                    ngd_dict = param_state['ngd_dict']
                    for axis in range(len(p.shape)):
                        d_p = ngd_dict[axis].precondition_directions(d_p)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-lr, d_p)

        return loss
