import jax.numpy as jnp
import flax.linen as nn

from jax.ops import segment_sum
from jax.nn.initializers import constant

from typing import (Any, Dict, Sequence)

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale, safe_mask
from mlff.nn.mlp import MLP
from mlff.nn.activation_function.activation_function import silu, softplus_inverse, softplus
import mlff.properties.property_names as pn
Array = Any


def sigma(x):
    return safe_mask(x > 0, fn=lambda u: jnp.exp(-1. / u), operand=x, placeholder=0)


def switching_fn(x, x_on, x_off):
    c = (x - x_on) / (x_off - x_on)
    return sigma(1 - c) / (sigma(1 - c) + sigma(c))


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    elif name == 'state_input_energy':
        return StateInputEnergy(**h)
    elif name == 'nac':
        return InterstateCoupling(**h)
    elif name == 'nac_potential':
        return InterstateCouplingPotential(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)


class Energy(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence = None
    per_atom_shift: Sequence = None
    num_embeddings: int = 100
    zbl_repulsion: bool = False
    zbl_repulsion_shift: float = 0.
    output_convention: str = 'per_structure'
    n_states: int = 1
    module_name: str = 'energy'

    def setup(self):
        self.energy_key = self.prop_keys[pn.energy]
        self.atomic_type_key = self.prop_keys[pn.atomic_type]
        if self.output_convention == 'per_atom':
            self.atomic_energy_key = self.prop_keys[pn.atomic_energy]

        def _take(arr):
            def fn(y, *args, **kwargs):
                taken = jnp.take(jnp.array(arr), y, axis=0)  # shape: (n) or (n,n_states)
                return taken if taken.ndim > 1 else taken[:, None]  # shape: (n,1) or (n,n_states)
            return fn

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = _take(self.per_atom_scale)
        else:
            self.get_per_atom_scale = lambda *args, **kwargs: jnp.float32(1.)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = _take(self.per_atom_shift)
        else:
            self.get_per_atom_shift = lambda *args, **kwargs: jnp.float32(0.)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']
        point_mask = inputs['point_mask']
        z = inputs[self.atomic_type_key].astype(jnp.int16)

        e_loc = MLP(features=[x.shape[-1], self.n_states], activation_fn=silu)(x)  # shape: (n,n_states)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n,n_states)
        e_loc = safe_scale(e_loc, scale=point_mask[:, None])  # shape: (n,n_states)

        if self.zbl_repulsion:
            e_rep = ZBLRepulsion(prop_keys=self.prop_keys,
                                 output_convention=self.output_convention)(inputs)  # shape: (n,1) or (1)
            e_rep = e_rep - jnp.asarray(self.zbl_repulsion_shift, dtype=e_rep.dtype)
        else:
            e_rep = jnp.asarray(0., dtype=e_loc.dtype)  # shape: (1)

        if self.output_convention == 'per_atom':
            return {self.atomic_energy_key: e_loc + e_rep}  # shape: (n,n_states)
        elif self.output_convention == 'per_structure':
            return {self.energy_key: e_loc.sum(axis=0) + e_rep}  # shape: (n_states)
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention
        # self.zbl_repulsion_energy.reset_output_convention(output_convention)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion': self.zbl_repulsion,
                                   'zbl_repulsion_shift': self.zbl_repulsion_shift,
                                   'n_states': self.n_states,
                                   'prop_keys': self.prop_keys}
                }


class StateInputEnergy(Energy):
    """
    Energy head of the *multi-state* variant: the electronic-state index is an input to the head,
    appended as it is to every atom's descriptor. For each state `s = 0, ..., n_states - 1` one
    per-atom MLP -- the same weights for every state -- reads `[x_i | s]`, the final invariant
    features of the So3krates layers with the raw state index as one extra float (no scaling, no
    embedding), and the per-atom outputs are summed per state.

    This is how the other multi-state models in our comparison take the state: MS-ANI appends the
    raw index to each atom's AEV, MS-NequIP to each atom's final scalar features. In So3krates the
    descriptor is `x_i` after the last layer. Because the state enters only here, the So3krates
    layers are state-independent and run once per structure; only this head runs once per state.

    Same fields as `Energy`, and the same output shapes as the multi-output `Energy(n_states=N)`:
    energy (n_states,) per structure, (n, n_states) per atom, so forces from `jax.jacrev`, the loss
    and the energy shift work unchanged. `per_atom_scale`/`per_atom_shift` are applied as in
    `Energy`, and ZBL repulsion, which does not depend on the state, is added to every state. The
    parameter count does not depend on `n_states`.

    Supersedes `StateEmbed` + `MultiStateStackNet` (a learned state embedding at the input and one
    full pass per state), which are kept so that checkpoints written with them still load.
    """
    module_name: str = 'state_input_energy'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        x = inputs['x']  # shape: (n,F)
        point_mask = inputs['point_mask']
        z = inputs[self.atomic_type_key].astype(jnp.int16)

        head = MLP(features=[x.shape[-1], 1], activation_fn=silu)  # one set of weights for all states
        e_loc = jnp.concatenate(
            [head(jnp.concatenate([x, jnp.full((x.shape[0], 1), float(s), dtype=x.dtype)], axis=-1))
             for s in range(self.n_states)], axis=-1)  # shape: (n,n_states)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n,n_states)
        e_loc = safe_scale(e_loc, scale=point_mask[:, None])  # shape: (n,n_states)

        if self.zbl_repulsion:
            e_rep = ZBLRepulsion(prop_keys=self.prop_keys,
                                 output_convention=self.output_convention)(inputs)  # shape: (n,1) or (1)
            e_rep = e_rep - jnp.asarray(self.zbl_repulsion_shift, dtype=e_rep.dtype)
        else:
            e_rep = jnp.asarray(0., dtype=e_loc.dtype)  # shape: (1)

        if self.output_convention == 'per_atom':
            return {self.atomic_energy_key: e_loc + e_rep}  # shape: (n,n_states)
        elif self.output_convention == 'per_structure':
            return {self.energy_key: e_loc.sum(axis=0) + e_rep}  # shape: (n_states)
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")


class InterstateCoupling(BaseSubModule):
    """
    Predicts the interstate (nonadiabatic) coupling vector between two electronic states.

    Rather than the raw coupling `d_01 = <psi_0|d/dR|psi_1>` (which diverges as the two
    states become degenerate), this predicts the smooth *scaled* coupling
    `h = d_01 * (E_1 - E_0)`, which has the same units/scale as a force and stays finite
    everywhere, including at conical intersections. Recover the raw coupling from `h` and a
    two-state energy prediction with `nn.nac_from_scaled_coupling`.

    The output is a per-atom 3-vector. `readout` selects how it is built:

    - `'chi'` (default, and what earlier checkpoints used): an invariant-gated copy of the
      backbone's own degree-1 equivariant feature (the l=1 block of `chi`, the SPHC
      representation already accumulated over all layers), `h_i = s * a(x_i) * chi_i[l=1]`.
      Because `a(x_i)` is an invariant scalar (computed from the invariant per-atom features
      `x`) and `chi_i[l=1]` is already a rotation-equivariant vector, `h_i` is
      rotation-equivariant and (since `chi` is built from relative pair directions)
      translation-invariant, with no additional Clebsch-Gordan machinery required. Requires
      `1` to be among the `degrees` the backbone was built with.

      **`chi` carries only one channel per degree**, so this form leaves the *direction* of
      `h_i` fully determined by the geometry -- the network can only choose a signed magnitude,
      i.e. one learnable degree of freedom per atom for a three-degree-of-freedom target. On
      this project's fulvene data that caps the achievable R^2 on `h` at ~0.97 (measured by
      projecting the reference coupling onto the trained model's own `chi[l=1]` directions),
      which is also roughly where a fully trained `'chi'` model lands.

    - `'pair'`: a message-passing vector readout that removes that cap,
      `h_i = s * sum_j c_ij * phi_r_cut_ij * unit_r_ij`, where `c_ij` is an invariant scalar
      from an MLP over `(x_i + x_j, x_i * x_j, rbf_ij)`. Since a real molecule has many
      neighbours per atom, `{unit_r_ij}_j` spans all of R^3 and the head can point `h_i`
      anywhere. Every ingredient of `c_ij` is *symmetric* under `i <-> j` by construction,
      while `unit_r_ji = -unit_r_ij`, so `sum_i h_i = 0` holds exactly -- the translational
      sum rule that a true nonadiabatic coupling obeys (this dataset's reference couplings
      satisfy it to ~0.4% median, whereas a trained `'chi'` head violates it by ~11%).

    - `'chi+pair'`: the sum of both terms, each with its own learnable scale.

    `enforce_sum_rule` additionally projects out any residual net translation
    (`h_i <- h_i - mean_j h_j` over the real, non-padded atoms). It is a no-op up to floating
    point for `'pair'`, and is what makes `'chi'`/`'chi+pair'` satisfy the sum rule too. It
    defaults to `False` so that checkpoints written before this option existed keep
    reproducing their original predictions when reloaded.

    `s` is a single learnable scalar (init `output_scale`). `chi`'s degree-1 block is
    normalized to a small internal SPHC scale (it is designed to be combined multiplicatively
    inside the attention layers, not read out directly), typically one to two orders of
    magnitude below a physical coupling target -- without `s`, fitting that gap would require
    the whole gate MLP (and, through it, the shared backbone) to grow large weights just to
    reach the right output scale, which is slow and poorly conditioned under Adam. A single
    extra scalar parameter lets the optimizer fix the overall magnitude in a handful of steps
    independently of learning the (much harder) directional/angular dependence.

    If `use_grad_diff` is set, an additional per-atom vector input (`prop_keys[pn.grad_diff]`,
    the gradient-difference vector dE_1/dR - dE_0/dR) is read and combined with the backbone
    term via its own independent invariant gate and learnable scale: `h_i = s*a(x_i)*chi_i[l=1]
    + s2*a2(x_i, |g_i|)*g_i`. `g` is already a genuine Cartesian per-atom vector (unlike `chi`'s
    l=1 block, which needs the reordering below), so it is used as-is; its norm is an invariant
    scalar fed into the second gate alongside `x`. This is intended as an oracle/diagnostic
    input (true ab-initio gradients, not a network prediction) to test whether gradient
    information helps NAC fitting near conical intersections before committing to a real
    gradient-predicting model to supply it at inference time.
    """
    prop_keys: Dict
    degrees: Sequence[int]
    output_scale: float = 1.
    module_name: str = 'nac'
    use_grad_diff: bool = False
    readout: str = 'chi'
    enforce_sum_rule: bool = False

    def setup(self):
        self.nac_key = self.prop_keys[pn.nac]
        if self.use_grad_diff:
            self.grad_diff_key = self.prop_keys[pn.grad_diff]
        if self.readout not in ('chi', 'pair', 'chi+pair'):
            msg = (f"`InterstateCoupling.readout` must be one of 'chi', 'pair', 'chi+pair'; "
                   f"got {self.readout!r}.")
            raise ValueError(msg)
        self._use_chi = 'chi' in self.readout
        self._use_pair = 'pair' in self.readout
        if self._use_chi:
            if 1 not in self.degrees:
                msg = (f"`InterstateCoupling` with readout={self.readout!r} reads the degree-1 "
                      f"(vector) block of `chi`, which requires `1` to be in `degrees`; got "
                      f"degrees={self.degrees}.")
                raise ValueError(msg)
            n0 = list(self.degrees).index(1)
            offset = sum(2 * d + 1 for d in self.degrees[:n0])
            self._l1_slice = slice(offset, offset + 3)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        x = inputs['x']  # shape: (n,F), invariant
        point_mask = inputs['point_mask']  # shape: (n)

        h = jnp.zeros((x.shape[0], 3), dtype=x.dtype)  # shape: (n,3), equivariant

        if self._use_chi:
            chi = inputs['chi']  # shape: (n,m_tot), equivariant
            chi_l1 = chi[:, self._l1_slice]  # shape: (n,3); real-SPHC order (m=-1,0,1) ~ (y,z,x)
            # `basis_function/spherical.py`'s l=1 real spherical harmonics are laid out as
            # (Y_1^-1, Y_1^0, Y_1^1) = c*(y, z, x) for a single shared constant c (see module
            # docstring) -- a fixed permutation of Cartesian (x,y,z), not (x,y,z) itself. Verified
            # empirically: comparing this block directly against a Cartesian-rotated reference gives
            # ~50-180% relative error (looks like broken equivariance), while reordering it to
            # (x,y,z) first reproduces the expected rotation to float32 precision (~1e-7). Since
            # (x,y,z) is the convention real NAC/force training data and `nn.nac_from_scaled_coupling`
            # use, this reordering is required for the *physical* Cartesian vector -- not optional,
            # and not merely a preference -- despite the un-reordered block already being a
            # perfectly valid (if differently-labeled) O(3) representation on its own.
            chi_l1 = chi_l1[:, jnp.array([2, 0, 1])]  # shape: (n,3), Cartesian (x,y,z)

            gate = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x)  # shape: (n,1), invariant
            s = self.param('output_scale', constant(self.output_scale), (1,))  # shape: (1), invariant
            h = h + s * gate * chi_l1  # shape: (n,3), equivariant

        if self._use_pair:
            idx_i = inputs[self.prop_keys[pn.idx_i]]  # shape: (P)
            idx_j = inputs[self.prop_keys[pn.idx_j]]  # shape: (P)
            pair_mask = inputs['pair_mask']  # shape: (P)
            rbf_ij = inputs['rbf_ij']  # shape: (P,K), invariant
            phi_r_cut = inputs['phi_r_cut']  # shape: (P), invariant
            unit_r_ij = inputs['unit_r_ij']  # shape: (P,3), equivariant

            x_i = x[idx_i]  # shape: (P,F)
            x_j = x[idx_j]  # shape: (P,F)
            # Both pair features are symmetric under i <-> j, and so are `rbf_ij`/`phi_r_cut`
            # (functions of the scalar distance). That symmetry is what makes `sum_i h_i`
            # vanish identically below, since `unit_r_ji = -unit_r_ij`.
            pair_in = jnp.concatenate([x_i + x_j, x_i * x_j, rbf_ij], axis=-1)  # shape: (P,2F+K)
            c_ij = MLP(features=[x.shape[-1], 1], activation_fn=silu)(pair_in)[:, 0]  # shape: (P)
            c_ij = safe_scale(c_ij * phi_r_cut, scale=pair_mask)  # shape: (P)

            s_pair = self.param('pair_scale', constant(self.output_scale), (1,))  # shape: (1)
            h_pair = segment_sum(c_ij[:, None] * unit_r_ij,
                                 segment_ids=idx_i,
                                 num_segments=x.shape[0])  # shape: (n,3), equivariant
            h = h + s_pair * h_pair

        if self.use_grad_diff:
            g = inputs[self.grad_diff_key]  # shape: (n,3), Cartesian, equivariant
            g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)  # shape: (n,1), invariant
            gate2 = MLP(features=[x.shape[-1], 1], activation_fn=silu)(
                jnp.concatenate([x, g_norm], axis=-1))  # shape: (n,1), invariant
            s2 = self.param('grad_diff_scale', constant(self.output_scale), (1,))  # shape: (1)
            h = h + s2 * gate2 * g  # shape: (n,3), equivariant

        h = safe_scale(h, scale=point_mask[:, None])  # shape: (n,3)

        if self.enforce_sum_rule:
            # Project out any net translation: a true coupling satisfies sum_i d_i = 0, because
            # rigidly translating the molecule leaves the electronic wavefunctions unchanged.
            # The mean is over real atoms only -- padded atoms already contribute zero above.
            n_real = point_mask.sum()  # shape: ()
            h = h - safe_mask(n_real > 0,
                              fn=lambda u: h.sum(axis=0, keepdims=True) / u,
                              operand=n_real,
                              placeholder=0.)  # shape: (n,3)
            h = safe_scale(h, scale=point_mask[:, None])  # shape: (n,3)

        return {self.nac_key: h}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': list(self.degrees),
                                   'output_scale': self.output_scale,
                                   'use_grad_diff': self.use_grad_diff,
                                   'readout': self.readout,
                                   'enforce_sum_rule': self.enforce_sum_rule,
                                   'prop_keys': self.prop_keys}}


class InterstateCouplingPotential(BaseSubModule):
    """
    Predicts an invariant scalar "coupling potential" `S(R)` whose gradient with respect to the
    atomic positions is used as the (scaled) interstate coupling: `h_i = dS/dR_i`.

    This is the same construction SchNet/SchNarc use for nonadiabatic couplings -- and the same
    one `mlff` already uses for forces (`nn.get_obs_and_force_fn`) -- applied to a "virtual"
    property that has no independent physical meaning of its own. The module itself only emits
    the scalar; the differentiation is wired up alongside the force derivative in
    `nn/stacknet/observable_function.py::get_obs_and_force_fn(nac_from_potential=True)`, so the
    `nac` output key is produced there, not here.

    Why it is worth trying: the resulting vector field is rotation-equivariant and
    translation-invariant by construction (it inherits both from the invariant scalar), it has
    the full three degrees of freedom per atom that a single-channel `chi[l=1]` readout lacks,
    and it satisfies the translational sum rule `sum_i h_i = 0` exactly, since `S` depends on
    the positions only through relative displacements.

    The caveat is that it also *imposes* a constraint the true coupling does not obey: any
    gradient field is curl-free in the 3N-dimensional configuration space, and the nonadiabatic
    coupling is not exactly a gradient of a scalar. So this trades a genuine loss of generality
    for exact symmetry and full directional freedom, and costs an extra reverse-mode pass per
    evaluation. Whether that trade pays off is an empirical question -- hence both this and
    `InterstateCoupling(readout='pair')` exist to be compared.
    """
    prop_keys: Dict
    output_scale: float = 1.
    module_name: str = 'nac_potential'

    def setup(self):
        self.nac_potential_key = self.prop_keys[pn.nac_potential]

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        x = inputs['x']  # shape: (n,F), invariant
        point_mask = inputs['point_mask']  # shape: (n)

        s_loc = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x)  # shape: (n,1), invariant
        s_loc = safe_scale(s_loc, scale=point_mask[:, None])  # shape: (n,1)
        s = self.param('output_scale', constant(self.output_scale), (1,))  # shape: (1)

        return {self.nac_potential_key: s * s_loc.sum(axis=0)}  # shape: (1)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'output_scale': self.output_scale,
                                   'prop_keys': self.prop_keys}}


class ZBLRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark repulsion.
    """
    prop_keys: dict
    output_convention: str = 'per_structure'
    module_name: str = 'energy_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    def setup(self):
        self.energy_key = self.prop_keys[pn.energy]
        self.atomic_type_key = self.prop_keys[pn.atomic_type]
        self.atomic_position_key = self.prop_keys[pn.atomic_position]

        if self.output_convention == 'per_atom':
            self.atomic_energy_key = self.prop_keys[pn.atomic_energy]

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        pair_mask = inputs['pair_mask']

        phi_r_cut_ij = inputs['phi_r_cut']

        d_ij = inputs['d_ij']  # shape: (P)
        z = inputs[self.atomic_type_key]  # shape: (n)
        zf = z.astype(d_ij.dtype)  # shape: (n)

        idx_i = inputs[self.prop_keys[pn.idx_i]]  # shape: (P)
        idx_j = inputs[self.prop_keys[pn.idx_j]]  # shape: (P)
        z_i = zf[idx_i]
        z_j = zf[idx_j]

        z_d_ij = safe_mask(mask=d_ij != 0,
                           operand=d_ij,
                           fn=lambda u: z_i * z_j / u,
                           placeholder=0.
                           )  # shape: (P)

        x = self.ke * phi_r_cut_ij * z_d_ij  # shape: (P)

        rzd = d_ij * (jnp.power(z_i, p) + jnp.power(z_j, p)) * d  # shape: (P)
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd)
        # shape: (P)

        w = switching_fn(d_ij, x_on=0, x_off=1.5)  # shape: (P)
        e_rep_edge = safe_scale(w * x * y, scale=pair_mask) / jnp.asarray(2, dtype=d_ij.dtype)  # shape: (P)

        if self.output_convention == 'per_atom':
            return segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=len(z))[:, None]  # shape: (n,1)
        elif self.output_convention == 'per_structure':
            return e_rep_edge.sum(axis=0)  # shape: (1)
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")


def make_e_rep_fn(prop_keys, ds, cutoff_fn='cosine_cutoff_fn', r_cut=5, mic=False):
    from mlff.masking.mask import safe_scale
    from mlff.cutoff_function.pbc import add_cell_offsets
    from mlff.cutoff_function import get_cutoff_fn
    import jax

    cut_fn = get_cutoff_fn(cutoff_fn)
    P = ds[0][prop_keys[pn.idx_i]].shape[-1]
    # n = ds[0][prop_keys[pn.atomic_type]].shape[-1]

    init_inputs = {'phi_r_cut': jnp.ones(P),
                   'd_ij': jnp.ones(P),
                   'pair_mask': jnp.zeros(P),
                   **jax.tree_map(lambda x: jnp.array(x[0, ...]), ds[0])}

    zbl_repulsion = ZBLRepulsion(prop_keys=prop_keys)
    p = zbl_repulsion.init(jax.random.PRNGKey(0), init_inputs)
    zbl_repulsion_fn = lambda u: zbl_repulsion.apply(p, u)

    def e_rep_fn(inputs):
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = (idx_i > -1).astype(jnp.float32)  # shape: (n_pairs)

        R = inputs[prop_keys[pn.atomic_position]]  # shape: (n,3)
        # Calculate pairwise distance vectors
        r_ij = safe_scale(jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j), scale=pair_mask[:, None])
        # shape: (n_pairs,3)

        # Apply minimal image convention if needed
        if mic:
            cell = inputs[prop_keys[pn.unit_cell]]  # shape: (3,3)
            cell_offsets = inputs[prop_keys[pn.cell_offset]]  # shape: (n_pairs,3)
            r_ij = add_cell_offsets(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets)  # shape: (n_pairs,3)

        # Scale pairwise distance vectors with pairwise mask
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])

        # Calculate pairwise distances
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        phi_r_cut = safe_scale(cut_fn(d_ij, r_cut=r_cut), scale=pair_mask)  # shape: (n_pairs)

        y = {'phi_r_cut': phi_r_cut, 'd_ij': d_ij, 'pair_mask': pair_mask, **inputs}

        return zbl_repulsion_fn(y)

    return e_rep_fn


def estimate_zbl_repulsion_contribution(prop_keys, ds, cutoff_fn, r_cut, mic=False):
    import jax
    import numpy as np

    batch_size = 100
    N_data = len(ds[0][prop_keys[pn.atomic_position]])
    batch_size = batch_size if batch_size < N_data else N_data
    B = N_data // batch_size
    e_rep_fn = make_e_rep_fn(prop_keys=prop_keys, ds=ds, cutoff_fn=cutoff_fn, r_cut=r_cut, mic=mic)
    e_rep_fn = jax.jit(jax.vmap(e_rep_fn))

    x = np.zeros(batch_size)
    for b in range(B):
        train_batch = jax.tree_map(lambda y: y[int(b*batch_size):int((b+1)*batch_size), ...], ds[0])
        x += np.array(e_rep_fn(train_batch), dtype=np.float64)

    x = x / B
    x = x.mean()
    return x
