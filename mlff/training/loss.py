import jax.numpy as jnp
from typing import (Callable, Dict, Sequence, Tuple)

from mlff.masking.mask import safe_mask
from mlff.properties import property_names as pn

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
BatchImportanceWeights = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]


def mse_loss(*, y, y_true): return jnp.mean((y_true - y)**2)


def scaled_safe_masked_mse_loss(y, y_true, scale, msk):
    """

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B,*)

    Returns:

    """
    full_mask = ~jnp.isnan(y_true) & msk
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    den = full_mask.reshape(-1).sum().astype(dtype=v.dtype)
    return safe_mask(den > 0, lambda x: v.reshape(-1).sum() / x, den, 0)


def _per_example_masked_mse(y, y_true, scale, msk):
    """
    Same masked, scaled squared error as `scaled_safe_masked_mse_loss`, but reduced only over
    the non-batch axes, leaving one value per example (shape: (B,)) instead of a single scalar.

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B,*)

    Returns: shape (B,)

    """
    full_mask = ~jnp.isnan(y_true) & msk
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    axes = tuple(range(1, v.ndim))
    v_sum = v.sum(axis=axes)
    count = full_mask.sum(axis=axes).astype(dtype=v.dtype)
    return safe_mask(count > 0, lambda c: v_sum / c, count, 0)


def _force_mask(u, target_ndim):
    """
    Node mask `u` has shape (B, n_atoms). The force target is (B, n_atoms, 3) in the single-state
    case, or (B, n_states, n_atoms, 3) for multi-state training. `extra` inserts the additional
    middle (state) axes so the mask broadcasts correctly against either shape.
    """
    extra = target_ndim - u.ndim - 1
    return u.reshape(u.shape[:1] + (1,) * extra + u.shape[1:] + (1,))


masks = {pn.energy: lambda u, target_ndim=None: jnp.ones(len(u)).astype(bool)[:, None],
         pn.atomic_energy: lambda u, target_ndim=None: u[..., None],
         pn.force: _force_mask,
         pn.nac: _force_mask,
         pn.stress: lambda u, target_ndim=None: jnp.ones(len(u)).astype(bool)[:, None, None],
         pn.partial_charge: lambda u, target_ndim=None: u[..., None],
         pn.hirshfeld_volume: lambda u, target_ndim=None: u[..., None],
         pn.hirshfeld_volume_ratio: lambda u, target_ndim=None: u[..., None]
         }


def get_loss_fn(obs_fn: Callable,
                weights: Dict,
                prop_keys: Dict,
                scales: Dict = None,
                gap_weight: float = None,
                sign_invariant_targets: Sequence[str] = ()):
    """
    Args:
        gap_weight (float, optional): if given, adds an extra MSE term on adjacent-state energy
            gaps (`E[..., s+1] - E[..., s]` for every `s`), on top of the per-property losses
            over `weights`. Meant for multi-state (`n_states > 1`) training, where independent
            per-state energy heads are not otherwise encouraged to have correlated errors, so the
            gap (predicted minus true) can be noisier than either state's own energy error --
            this term supervises the gap directly. No-op if `pn.energy` is not in `weights`, or
            if the energy target only has one state (nothing to take a gap of).
        sign_invariant_targets (Sequence[str], optional): property names (e.g. `pn.nac`) whose
            per-property loss should be, for every example in the batch independently,
            `min(loss(y, y_true), loss(y, -y_true))` instead of `loss(y, y_true)`. Meant for
            targets with an arbitrary, per-example sign/phase ambiguity -- e.g. nonadiabatic
            coupling vectors, whose sign depends on an arbitrary electronic-wavefunction phase
            convention that is generally not consistent across independently-computed
            geometries (the whole per-atom vector field for one geometry shares one arbitrary
            sign, since the phase convention is a single choice per single-point calculation).
            A plain MSE against such a target fights an essentially random sign on top of the
            real signal; this makes the loss agnostic to that overall sign, per example. The
            sign choice is resolved per example, not once for the whole batch -- with a roughly
            even mix of + and - signed examples in a batch, a single batch-wide sign choice
            would leave the loss unchanged from plain MSE, so this must not be aggregated
            across the batch before taking the min. No-op for targets not in this sequence.
    """
    _weights = {prop_keys[k]: v for (k, v) in weights.items()}
    if scales is None:
        _scales = {prop_keys[k]: jnp.ones(1) for (k, _) in weights.items()}
    else:
        _scales = scales

    _masks = {prop_keys[k]: masks[k] for k in weights.keys()}
    _energy_key = prop_keys.get(pn.energy)
    _sign_invariant_keys = {prop_keys[k] for k in sign_invariant_targets if k in weights}

    # _with_stress = pn.stress in list(weights.keys())

    def loss_fn(params, batch: DataTupleT):
        inputs, targets = batch
        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():  # name is the value in prop_keys
            msk = _masks[name](inputs[prop_keys[pn.node_mask]], targets[name].ndim)
            if name in _sign_invariant_keys:
                _err_pos = _per_example_masked_mse(y=outputs[name], y_true=target, scale=_scales[name], msk=msk)
                _err_neg = _per_example_masked_mse(y=outputs[name], y_true=-target, scale=_scales[name], msk=msk)
                _l = jnp.minimum(_err_pos, _err_neg).mean()
            else:
                _l = scaled_safe_masked_mse_loss(y=outputs[name], y_true=target, scale=_scales[name], msk=msk)

            loss += _weights[name] * _l
            train_metrics.update({name: _l / _scales[name].mean()})

        if gap_weight is not None and _energy_key is not None and _energy_key in targets \
                and targets[_energy_key].shape[-1] > 1:
            e_pred, e_true = outputs[_energy_key], targets[_energy_key]  # shape (B, n_states)
            gap_pred = e_pred[..., 1:] - e_pred[..., :-1]  # adjacent-state gaps: (B, n_states - 1)
            gap_true = e_true[..., 1:] - e_true[..., :-1]
            gap_mask = _masks[_energy_key](inputs[prop_keys[pn.node_mask]], e_true.ndim)
            _gap_l = scaled_safe_masked_mse_loss(y=gap_pred, y_true=gap_true, scale=jnp.ones(1), msk=gap_mask)
            loss += gap_weight * _gap_l
            train_metrics.update({'gap': _gap_l})

        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics

    return loss_fn


def get_active_learning_loss_fn(obs_fn, weights):
    def loss_fn(params, batch: BatchImportanceWeights):
        inputs, targets, imp_weights = batch  # shape: (B, ...), (B, ...), (B)
        imp_weights_normalized = jnp.sqrt(imp_weights / imp_weights.mean())  # shape: (B)

        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():
            _y = jnp.einsum('b, b... -> b...', imp_weights_normalized, outputs[name])
            _y_true = jnp.einsum('b, b... -> b...', imp_weights_normalized, targets[name])
            _l = mse_loss(y=_y, y_true=_y_true)
            loss += weights[name] * _l
            train_metrics.update({name: _l})
        # for name, w in weights.items():
        #     _l = mse_loss(y_true=outputs[name], y=targets[name])
        #     loss += w * _l
        #     train_metrics.update({name: _l})
        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics
    return loss_fn
