import jax
import jax.numpy as jnp
import flax.linen as nn
import logging

from functools import partial
from typing import (Any, Dict, Sequence)
from jax.ops import segment_sum

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_mask
from mlff.masking.mask import safe_scale
from mlff.basis_function.radial import get_rbf_fn
from mlff.cutoff_function import add_cell_offsets
from mlff.cutoff_function import get_cutoff_fn
from mlff.basis_function.spherical import init_sph_fn
from mlff.properties import property_names as pn


# TODO: write init_from_dict methods in order to improve backward compatibility. E.g. AtomTypeEmbed(**h)
# will only work as long as the non-default properties of the class are exactly the ones equal to the ones in h. As soon
# as additional arguments appear in h. Maybe use something like kwargs to allow for extensions?


class GeometryEmbed(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_function: str
    n_rbf: int
    radial_cutoff_fn: str
    r_cut: float
    sphc: bool
    sphc_normalization: float = None
    mic: bool = False
    solid_harmonic: bool = False
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed'

    def setup(self):
        if self.input_convention == 'positions':
            self.atomic_position_key = self.prop_keys.get('atomic_position')
            if self.mic == 'bins':
                logging.warning(f'mic={self.mic} is deprecated in favor of mic=True.')
            if self.mic == 'naive':
                raise DeprecationWarning(f'mic={self.mic} is not longer supported.')
            if self.mic:
                self.unit_cell_key = self.prop_keys.get('unit_cell')
                self.cell_offset_key = self.prop_keys.get('cell_offset')

        elif self.input_convention == 'displacements':
            self.displacement_vector_key = self.prop_keys.get('displacement_vector')
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        self.atomic_type_key = self.prop_keys.get('atomic_type')

        self.sph_fns = [init_sph_fn(y) for y in self.degrees]

        _rbf_fn = get_rbf_fn(self.radial_basis_function)
        self.rbf_fn = _rbf_fn(n_rbf=self.n_rbf, r_cut=self.r_cut)

        _cut_fn = get_cutoff_fn(self.radial_cutoff_fn)
        self.cut_fn = partial(_cut_fn, r_cut=self.r_cut)
        self._lambda = jnp.float32(self.sphc_normalization)

    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.
        Args:
            inputs (Dict):
                R (Array): atomic positions, shape: (n,3)
                idx_i (Array): index centering atom, shape: (n_pairs)
                idx_j (Array): index neighboring atom, shape: (n_pairs)
                pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:
        """
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = inputs['pair_mask']  # shape: (n_pairs)

        # depending on the input convention, calculate the displacement vectors or load them from input
        if self.input_convention == 'positions':
            R = inputs[self.atomic_position_key]  # shape: (n,3)
            # Calculate pairwise distance vectors
            r_ij = safe_scale(jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j), scale=pair_mask[:, None])
            # shape: (n_pairs,3)

            # Apply minimal image convention if needed
            if self.mic:
                cell = inputs[self.unit_cell_key]  # shape: (3,3)
                cell_offsets = inputs[self.cell_offset_key]  # shape: (n_pairs,3)
                r_ij = add_cell_offsets(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets)  # shape: (n_pairs,3)

        elif self.input_convention == 'displacements':
            R = None
            r_ij = inputs[self.displacement_vector_key]
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Scale pairwise distance vectors with pairwise mask
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])

        # Calculate pairwise distances
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        # Gaussian basis expansion of distances
        rbf_ij = safe_scale(self.rbf_fn(d_ij[:, None]), scale=pair_mask[:, None])  # shape: (n_pairs,K)
        phi_r_cut = safe_scale(self.cut_fn(d_ij), scale=pair_mask)  # shape: (n_pairs)

        # Normalized distance vectors
        unit_r_ij = safe_mask(mask=d_ij[:, None] != 0,
                              operand=r_ij,
                              fn=lambda y: y / d_ij[:, None],
                              placeholder=0
                              )  # shape: (n_pairs, 3)
        unit_r_ij = safe_scale(unit_r_ij, scale=pair_mask[:, None])  # shape: (n_pairs, 3)

        # Spherical harmonics
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            sph_ij = safe_scale(sph_fn(unit_r_ij), scale=pair_mask[:, None])  # shape: (n_pairs,2l+1)
            sph_harms_ij += [sph_ij]  # len: |L| / shape: (n_pairs,2l+1)

        sph_harms_ij = jnp.concatenate(sph_harms_ij, axis=-1) if len(self.degrees) > 0 else None
        # shape: (n_pairs,m_tot)

        geometric_data = {'R': R,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'rbf_ij': rbf_ij,
                          'phi_r_cut': phi_r_cut,
                          'sph_ij': sph_harms_ij,
                          }

        # Spherical harmonic coordinates (SPHCs)
        if self.sphc:
            z = inputs[self.atomic_type_key]
            point_mask = inputs['point_mask']
            if self.sphc_normalization is None:
                # Initialize SPHCs to zero
                geometric_data.update(_init_sphc_zeros(z=z,
                                                       sph_ij=sph_harms_ij,
                                                       phi_r_cut=phi_r_cut,
                                                       idx_i=idx_i,
                                                       point_mask=point_mask,
                                                       mp_normalization=self._lambda)
                                      )
            else:
                # Initialize SPHCs with a neighborhood dependent embedding
                geometric_data.update(_init_sphc(z=z,
                                                 sph_ij=sph_harms_ij,
                                                 phi_r_cut=phi_r_cut,
                                                 idx_i=idx_i,
                                                 point_mask=point_mask,
                                                 mp_normalization=self._lambda)
                                      )

        # Solid harmonics (Spherical harmonics + radial part)
        if self.solid_harmonic:
            rbf_ij = safe_scale(rbf_ij, scale=phi_r_cut[:, None])  # shape: (n_pairs,K)
            g_ij = sph_harms_ij[:, :, None] * rbf_ij[:, None, :]  # shape: (n_pair,m_tot,K)
            g_ij = safe_scale(g_ij, scale=pair_mask[:, None, None], placeholder=0)  # shape: (n_pair,m_tot,K)
            geometric_data.update({'g_ij': g_ij})

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_function': self.radial_basis_function,
                                   'n_rbf': self.n_rbf,
                                   'radial_cutoff_fn': self.radial_cutoff_fn,
                                   'r_cut': self.r_cut,
                                   'sphc': self.sphc,
                                   'sphc_normalization': self.sphc_normalization,
                                   'solid_harmonic': self.solid_harmonic,
                                   'mic': self.mic,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, point_mask, mp_normalization, *args, **kwargs):
    _sph_harms_ij = safe_scale(sph_ij, phi_r_cut[:, None])  # shape: (n_pairs,m_tot)
    chi = segment_sum(_sph_harms_ij, segment_ids=idx_i, num_segments=len(z))
    chi = safe_scale(chi, scale=point_mask[:, None])  # shape: (n,m_tot)
    return {'chi': chi / mp_normalization}


def _init_sphc_zeros(z, sph_ij, *args, **kwargs):
    return {'chi': jnp.zeros((z.shape[-1], sph_ij.shape[-1]), dtype=sph_ij.dtype)}


class AtomTypeEmbed(BaseSubModule):
    features: int
    prop_keys: Dict
    num_embeddings: int = 100
    module_name: str = 'atom_type_embed'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                z (Array): atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (n,F)

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)  # shape: (n)
        return safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z),
                          scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}


class StateEmbed(BaseSubModule):
    """
    **Superseded by `nn.StateInputEnergy`** (the raw state index appended to the descriptor at the
    energy head, as in MS-ANI and MS-NequIP), which new multi-state models use. Kept so that
    checkpoints written with it still load and predict as before.

    Embed the electronic-state index into the invariant atomic features, which is what makes the
    *multi-state* variant of the network multi-state: one weight-shared network is evaluated once
    per state with the state as an input, instead of a single pass emitting all states from
    independent output heads (`Energy`'s `n_states`, the multi-output variant).

    This is the So3krates form of MS-ANI's construction (10.26434/chemrxiv-2024-dtc1w), whose
    `StateInputNet` concatenates the state index as one extra scalar onto every atom's descriptor
    before the element-wise MLP. Concatenating a scalar `s` and applying a dense layer is
    `W @ descriptor + w * s`, so the equivalent here -- where the invariant features come from a
    learned atom-type embedding rather than a hand-built descriptor -- is to add `s * w` with a
    single learned direction `w`, shared by every atom of the structure. Consequences worth knowing:

      * The state enters as a *continuous* scalar, so nothing structurally prevents asking a model
        trained on two states for a third; whether the answer means anything is another matter.
      * State 0 contributes exactly nothing (`s = 0`), exactly as in MS-ANI.
      * `StackNet` sums its feature embeddings and divides by sqrt(n_embeds), so adding this module
        also rescales the atom-type embedding by 1/sqrt(2) -- a change of feature scale only, and
        the same for every state.
    """
    features: int
    prop_keys: Dict
    module_name: str = 'state_embed'

    def setup(self):
        self.state_key = self.prop_keys.get(pn.state)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Args:
            inputs (Dict):
                state (Array): electronic-state index, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args (Tuple):
            **kwargs (Dict):

        Returns: State embedding, broadcast over atoms, shape: (n,F)

        """
        s = jnp.asarray(inputs[self.state_key], dtype=jnp.float32).reshape(-1)[0]  # shape: ()
        point_mask = inputs['point_mask']

        w = self.param('state_direction', nn.initializers.normal(stddev=1.), (self.features,))
        x = jnp.broadcast_to(s * w[None, :], (point_mask.shape[0], self.features))  # shape: (n,F)
        return safe_scale(x, scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'features': self.features,
                                   'prop_keys': self.prop_keys}}


def _l1_slice_of(degrees):
    """Where the degree-1 (vector) block sits inside a concatenated SPHC vector."""
    if 1 not in degrees:
        raise ValueError(f"a degree-1 (vector) block is required; got degrees={degrees}")
    n0 = list(degrees).index(1)
    offset = sum(2 * d + 1 for d in degrees[:n0])
    return slice(offset, offset + 3)


# `basis_function/spherical.py`'s real l=1 harmonics are ordered (Y_1^-1, Y_1^0, Y_1^1) ~ (y,z,x),
# a fixed permutation of Cartesian (x,y,z). `InterstateCoupling` reads chi's l=1 block out with
# [2,0,1] to get (x,y,z); writing a Cartesian vector *into* that block needs the inverse.
_CARTESIAN_TO_SPHC_L1 = (1, 2, 0)


class GradDiffSPHCEmbed(BaseSubModule):
    """
    Add the gradient-difference vector `g_i = dE_1/dR_i - dE_0/dR_i` into the degree-1 block of the
    SPHCs, so that the *backbone* is conditioned on it rather than only the coupling readout.

    Why this exists. `InterstateCoupling(use_grad_diff=True)` consumes `g` at the very end, as one
    more additive term `s2 * a(x_i, |g_i|) * g_i` on the predicted coupling. That can only push the
    prediction *along* `g` -- and on this project's fulvene data the true coupling is close to
    orthogonal to the gradient difference (median |cos| 0.08-0.17 across the three test sets;
    random level in 3N=36 dimensions is ~0.13), so that direction is nearly useless. Measured on
    the trained `base`/`chipair` checkpoints, the term carries 33-37% of the predicted coupling's
    norm at |cos| 0.15-0.22 with the true target, i.e. magnitude in a near-random direction that
    the other terms then have to cancel. Meanwhile `g` never reaches the representation at all:
    only the scalar `|g_i|` enters that gate, and the gate multiplies only the `g` term, so the
    pair coefficients and the chi gate are blind to it.

    Feeding it into `chi` instead puts it where the network can actually use it. Every
    `So3kratesLayer` builds its invariant pair features from l=0 contractions of `chi_i - chi_j`
    (`d_gamma`), and the InteractionBlock mixes SPHC norms back into the invariant features `x`, so
    after one layer the attention coefficients, the pair coefficients `c_ij` and the readout gates
    are all functions of `g` -- through the invariants `g_i . g_j`, `g_i . r_ij`, `|g_i|` that those
    contractions generate. That is the equivariant analogue of handing a kernel method the whole
    gradient-difference vector as a descriptor, which on this data is worth roughly a factor of six
    in RMSE on the coupling.

    Only the *direction* goes in here: per-atom `|g_i|` spans two orders of magnitude across the
    dataset (p10 0.07, p99 8.9 eV/Angstrom on Testset2), so adding `g` raw would let a handful of
    near-degenerate geometries dominate `chi`. The magnitude is fed separately, and on a log scale,
    by `GradDiffEmbed` into the invariant features. `scale` is learnable and starts small, so
    training begins close to the model without this input and grows the term if it helps.
    """
    prop_keys: Dict
    degrees: Sequence[int]
    scale_init: float = 0.1
    module_name: str = 'grad_diff_sphc_embed'

    def setup(self):
        self.grad_diff_key = self.prop_keys[pn.grad_diff]
        self._l1 = _l1_slice_of(self.degrees)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> Dict:
        chi = inputs['chi']  # shape: (n,m_tot), equivariant
        point_mask = inputs['point_mask']  # shape: (n)
        g = inputs[self.grad_diff_key]  # shape: (n,3), Cartesian, equivariant

        g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)  # shape: (n,1), invariant
        g_hat = safe_mask(mask=g_norm != 0, operand=g, fn=lambda u: u / g_norm, placeholder=0.)
        g_hat = g_hat[:, jnp.array(_CARTESIAN_TO_SPHC_L1)]  # shape: (n,3), real-SPHC (y,z,x) order
        g_hat = safe_scale(g_hat, scale=point_mask[:, None])

        s = self.param('grad_diff_sphc_scale', nn.initializers.constant(self.scale_init), (1,))
        chi = chi.at[:, self._l1].add(s * g_hat)
        return {'chi': safe_scale(chi, scale=point_mask[:, None])}

    def __dict_repr__(self):
        return {self.module_name: {'degrees': list(self.degrees),
                                   'scale_init': self.scale_init,
                                   'prop_keys': self.prop_keys}}


class GradDiffEmbed(BaseSubModule):
    """
    Embed the *magnitude* of the per-atom gradient difference `|g_i|` into the invariant atomic
    features, the companion to `GradDiffSPHCEmbed`'s direction-only injection.

    `log1p(|g_i|)` rather than `|g_i|`: the raw norm spans p10 0.07 to p99 8.9 eV/Angstrom on this
    project's fulvene data, a factor of ~130, which is badly conditioned as a direct network input;
    the log compresses that to ~0.07-2.3 while staying monotone and exactly 0 at `|g| = 0`.

    `StackNet` sums its feature embeddings and divides by sqrt(n_embeds), so adding this module
    also rescales the atom-type embedding -- a change of feature scale only, identical for every
    atom, exactly as for `StateEmbed`.
    """
    features: int
    prop_keys: Dict
    module_name: str = 'grad_diff_embed'

    def setup(self):
        self.grad_diff_key = self.prop_keys[pn.grad_diff]

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        point_mask = inputs['point_mask']  # shape: (n)
        g = inputs[self.grad_diff_key]  # shape: (n,3)

        g_norm = jnp.linalg.norm(g, axis=-1, keepdims=True)  # shape: (n,1), invariant
        u = jnp.log1p(g_norm)  # shape: (n,1)
        y = nn.Dense(self.features)(nn.silu(nn.Dense(self.features)(u)))  # shape: (n,F)
        return safe_scale(y, scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'features': self.features,
                                   'prop_keys': self.prop_keys}}


class OneHotEmbed(BaseSubModule):
    prop_keys: Dict
    atomic_types: Sequence[int]
    module_name: str = 'one_hot_embed'

    def setup(self):
        self.to_one_hot = lambda x: to_onehot(x, node_types=self.atomic_types)
        self.atomic_type_key = self.prop_keys[pn.atomic_type]

    def __call__(self, inputs: Dict, *args, **kwargs):
        z = inputs[self.atomic_type_key]
        return {'z_one_hot': self.to_one_hot(z)}

    def __dict_repr__(self):
        return {self.module_name: {'atomic_types': self.atomic_types,
                                   'prop_keys': self.prop_keys}
                }

    def reset_input_convention(self, input_convention: str) -> None:
        pass


def to_onehot(features: jnp.ndarray, node_types: Sequence):
    r"""
    Create onehot encoded vectors from :data:`features`.

    Args:
        features (Array): type of the nodes, shape: (n)
        node_types (Sequence[int]): list of possible node types.
    """
    ones = []
    for i, e in enumerate(node_types):
        ones.append(jnp.where(features == e, jnp.ones(1), jnp.zeros(1))[..., None])
    return jnp.concatenate(ones, axis=-1)