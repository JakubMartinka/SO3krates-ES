"""
Tests for `nn.InterstateCoupling` (nonadiabatic/interstate coupling vector prediction) and its
supporting loss/utility machinery.

Beyond shape/registration sanity checks, this exercises two properties that are easy to get
architecturally "correct-looking" but wrong in practice:

- Equivariance (`test_interstate_coupling_is_rotation_equivariant`): the predicted vector must
  rotate exactly like a real per-atom vector observable would.
- Actual learning ability (`test_interstate_coupling_learns_synthetic_equivariant_target`,
  `test_sign_invariant_loss_recovers_phase_scrambled_target`): fitting a genuine equivariant
  vector field, and specifically recovering a field whose sign is scrambled per example (as real
  ab-initio nonadiabatic coupling vectors are -- see `training.loss.get_loss_fn`'s
  `sign_invariant_targets` docstring). All numeric thresholds below were calibrated against
  actual runs of this exact configuration (fixed seeds throughout -> fully deterministic), not
  guessed, since both "does it learn" and "does the sign-invariant loss actually help" are
  empirical claims that can look right on paper and still fail in practice (an earlier version
  of the sign-invariant loss aggregated the +/- comparison over the whole batch instead of per
  example, which made it silently identical to plain MSE -- these tests would have caught that).
"""
import numpy as np
import pytest

from .test_data import load_data


def _nac_prop_keys():
    from mlff.properties import md17_property_keys
    return dict(md17_property_keys)


def _build_net(prop_keys, F=32, n_layer=2, degrees=(1, 2), n_heads=1, n_states=1):
    from mlff import nn

    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states),
          nn.InterstateCoupling(prop_keys=prop_keys, degrees=list(degrees))]
    return nn.So3krates(F=F, n_layer=n_layer, prop_keys=prop_keys, obs=obs,
                        geometry_embed_kwargs={'degrees': list(degrees)},
                        so3krates_layer_kwargs={'n_heads': n_heads, 'degrees': list(degrees)})


def test_interstate_coupling_dict_repr():
    from mlff import nn

    prop_keys = _nac_prop_keys()
    net = _build_net(prop_keys)

    h = net.__dict_repr__()
    stack_net_obs = h['stack_net']['observables']
    names = [list(o.keys())[0] for o in stack_net_obs]
    assert 'nac' in names

    nac_repr = stack_net_obs[names.index('nac')]['nac']
    assert nac_repr['degrees'] == [1, 2]

    # round-trips through the same registry StackNet.create_from_ckpt_dir uses
    from mlff.nn.observable import get_observable_module
    mod = get_observable_module('nac', nac_repr)
    assert isinstance(mod, nn.InterstateCoupling)
    assert mod.degrees == [1, 2]


def test_interstate_coupling_requires_degree_one():
    import jax
    import jax.numpy as jnp
    from mlff import nn

    prop_keys = _nac_prop_keys()
    mod = nn.InterstateCoupling(prop_keys=prop_keys, degrees=[2, 3])
    inputs = {'x': jnp.ones((3, 4)),
             'chi': jnp.ones((3, 12)),
             'point_mask': jnp.ones(3)}

    with pytest.raises(ValueError):
        mod.init(jax.random.PRNGKey(0), inputs)


def test_interstate_coupling_forward_shapes():
    import jax
    import jax.numpy as jnp
    import mlff.properties.property_names as pn

    prop_keys = _nac_prop_keys()
    n_atoms = 7
    net = _build_net(prop_keys, F=36, degrees=(1, 2, 3), n_heads=4, n_states=2)

    inputs = {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(0), shape=(n_atoms, 3)),
             prop_keys[pn.atomic_type]: jnp.ones(n_atoms),
             prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
             prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}

    params = net.init(jax.random.PRNGKey(0), inputs)
    out = net.apply(params, inputs)

    assert out[prop_keys[pn.nac]].shape == (n_atoms, 3)
    assert out[prop_keys[pn.energy]].shape == (2,)


def test_nac_from_scaled_coupling_roundtrip():
    import jax.numpy as jnp
    from mlff.nn import nac_from_scaled_coupling

    nac = jnp.array([[1.0, 0.0, -2.0], [0.5, 0.5, 0.5]])  # shape: (n_atoms, 3)
    gap = jnp.array(0.2)
    energy = jnp.array([0.0, 0.2])  # E_1 - E_0 == gap
    h = nac * gap

    recovered = nac_from_scaled_coupling(h, energy)
    assert np.allclose(np.asarray(recovered), np.asarray(nac), atol=1e-5)

    # near-degenerate gap: must not blow up / NaN, and should stay bounded by the eps floor
    tiny_gap_energy = jnp.array([0.0, 1e-8])
    h_tiny = nac * 1e-8
    recovered_tiny = nac_from_scaled_coupling(h_tiny, tiny_gap_energy, eps=1e-3)
    assert np.all(np.isfinite(np.asarray(recovered_tiny)))
    assert np.max(np.abs(np.asarray(recovered_tiny))) <= np.max(np.abs(np.asarray(nac))) * (1e-8 / 1e-3) + 1e-6


def test_interstate_coupling_is_rotation_equivariant():
    import jax
    import jax.numpy as jnp
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import property_names as pn
    from mlff.data import DataSet, DataTuple
    from mlff.geometric import get_rotation_matrix, apply_rotation

    prop_keys = _nac_prop_keys()
    net = _build_net(prop_keys)

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.jit(jax.vmap(obs_fn, in_axes=(None, 0)))

    data = dict(load_data('ethanol.npz'))
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=2, n_valid=1, n_test=None, mic=False, r_cut=5, training=True, seed=0)
    d = data_set.get_data_split()

    data_tuple = DataTuple(inputs=[pn.atomic_position, pn.atomic_type, pn.node_mask, pn.idx_i, pn.idx_j],
                           targets=[pn.energy, pn.force],
                           prop_keys=prop_keys)
    ds = data_tuple(d['train'])

    params = net.init(jax.random.PRNGKey(0), jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), ds[0]))
    inputs = jax.tree_util.tree_map(lambda x: jnp.array(x), ds[0])

    nac_key = prop_keys[pn.nac]
    base = obs_fn(params, inputs)
    NAC_base = base[nac_key]

    # the head must produce a genuinely non-trivial (non-zero) vector to make the equivariance
    # check below meaningful -- a head that is (buggily) always zero would trivially "pass" it.
    assert float(jnp.abs(NAC_base).max()) > 0

    for _ in range(5):
        M_rot = get_rotation_matrix(euler_axes='xyz', angles=np.random.rand(3) * 360, degrees=True)
        inputs_rot = {k: v for (k, v) in inputs.items()}
        inputs_rot['R'] = apply_rotation(inputs['R'], m_rot=M_rot)

        NAC_rot = obs_fn(params, inputs_rot)[nac_key]

        # rotating the input must change the raw output ...
        assert ~np.isclose(NAC_rot.reshape(-1), NAC_base.reshape(-1)).all()
        # ... exactly by the same rotation applied to the un-rotated output (equivariance).
        assert np.isclose(apply_rotation(NAC_base, M_rot).reshape(-1), NAC_rot.reshape(-1), atol=1e-5).all()


def _full_batch_train(net, loss_fn, train_batch, valid_batch, n_steps, lr, seed):
    import jax
    import jax.numpy as jnp
    import optax

    example = jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), train_batch[0])
    params = net.init(jax.random.PRNGKey(seed), example)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = tx.init(params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    (initial_loss, _), _ = grad_fn(params, valid_batch)
    for _ in range(n_steps):
        (_, _), grads = grad_fn(params, train_batch)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    (final_loss, _), _ = grad_fn(params, valid_batch)

    return params, float(initial_loss), float(final_loss)


def _synthetic_h_dataset(seed=0, scramble_sign=False):
    """
    A genuine (non-trivial) rotation-equivariant, translation-invariant per-atom vector field
    derived from real ethanol geometries: each atom's displacement from its structure's
    centroid. Standing in for a real scaled-coupling target `h` in tests that don't have access
    to actual ab-initio nonadiabatic coupling data, while still requiring the network to learn a
    real geometry -> vector mapping (not e.g. a constant).
    """
    import jax
    import jax.numpy as jnp
    from mlff.data import DataSet, DataTuple
    import mlff.properties.property_names as pn

    prop_keys = _nac_prop_keys()
    nac_key = prop_keys[pn.nac]

    data = dict(load_data('ethanol.npz'))
    R = np.asarray(data[prop_keys[pn.atomic_position]])  # (n_data, n_atoms, 3)
    com = R.mean(axis=1, keepdims=True)
    true_h = (0.1 * (R - com)).astype(np.float64)  # (n_data, n_atoms, 3)

    if scramble_sign:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1.0, 1.0], size=len(true_h)).astype(np.float64)
        data[nac_key] = true_h * signs[:, None, None]
    else:
        data[nac_key] = true_h

    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=40, n_valid=20, n_test=None, r_cut=5, training=True, seed=seed)
    d = data_set.get_data_split()

    data_tuple = DataTuple(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask],
                           targets=[pn.nac],
                           prop_keys=prop_keys)
    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    # ground truth for the validation split, re-derived with a *consistent* sign, regardless of
    # `scramble_sign` -- this is what an evaluation against real (phase-corrected) data would
    # look like, and is what the phase-invariance test below checks predictions against.
    valid_R = np.asarray(d['valid'][prop_keys[pn.atomic_position]])
    valid_true_h = 0.1 * (valid_R - valid_R.mean(axis=1, keepdims=True))

    train_batch = (jax.tree_util.tree_map(jnp.array, train_ds[0]), jax.tree_util.tree_map(jnp.array, train_ds[1]))
    valid_batch = (jax.tree_util.tree_map(jnp.array, valid_ds[0]), jax.tree_util.tree_map(jnp.array, valid_ds[1]))

    return prop_keys, train_batch, valid_batch, valid_true_h


def test_interstate_coupling_learns_synthetic_equivariant_target():
    """
    Sanity check that gradients actually flow through `InterstateCoupling` into a real fit, not
    just that shapes line up: train on a consistently-signed synthetic target and check the
    held-out loss drops substantially from its (untrained) initial value.
    """
    import jax
    from mlff.nn import get_obs_and_force_fn
    from mlff.training.loss import get_loss_fn
    import mlff.properties.property_names as pn

    prop_keys, train_batch, valid_batch, _ = _synthetic_h_dataset(seed=0, scramble_sign=False)
    net = _build_net(prop_keys, F=32, n_layer=2, degrees=(1, 2), n_heads=1)
    obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(net), in_axes=(None, 0)))
    loss_fn = get_loss_fn(obs_fn=obs_fn, weights={pn.nac: 1.0}, prop_keys=prop_keys)

    _, initial_loss, final_loss = _full_batch_train(net, loss_fn, train_batch, valid_batch,
                                                    n_steps=300, lr=1e-2, seed=0)

    # empirically (fixed seeds -> deterministic): initial ~0.0081, final ~0.0050 (~40% drop).
    # 0.75x leaves a comfortable margin without demanding full convergence from a tiny net/short run.
    assert final_loss < 0.75 * initial_loss


def test_sign_invariant_loss_recovers_phase_scrambled_target():
    """
    The core claim behind `sign_invariant_targets`: real nonadiabatic coupling data has an
    arbitrary, per-geometry sign (electronic-wavefunction phase) that is not consistent across
    independently-computed geometries (confirmed empirically for this project's own NAC data --
    see CLAUDE.md). This trains two otherwise-identical models on the *same* phase-scrambled
    target -- one with plain MSE, one with `sign_invariant_targets=(pn.nac,)` -- and checks that
    only the sign-invariant one actually recovers the underlying (consistently-signed) field.
    """
    import jax
    from mlff.nn import get_obs_and_force_fn
    from mlff.training.loss import get_loss_fn
    import mlff.properties.property_names as pn

    prop_keys, train_batch, valid_batch, valid_true_h = _synthetic_h_dataset(seed=0, scramble_sign=True)
    nac_key = prop_keys[pn.nac]

    def sign_invariant_mae(pred, true):
        d_pos = np.abs(pred - true).mean(axis=(1, 2))
        d_neg = np.abs(pred + true).mean(axis=(1, 2))
        return float(np.minimum(d_pos, d_neg).mean())

    zero_baseline = sign_invariant_mae(np.zeros_like(valid_true_h), valid_true_h)

    net_plain = _build_net(prop_keys, F=32, n_layer=2, degrees=(1, 2), n_heads=1)
    obs_fn_plain = jax.jit(jax.vmap(get_obs_and_force_fn(net_plain), in_axes=(None, 0)))
    loss_fn_plain = get_loss_fn(obs_fn=obs_fn_plain, weights={pn.nac: 1.0}, prop_keys=prop_keys)
    params_plain, _, _ = _full_batch_train(net_plain, loss_fn_plain, train_batch, valid_batch,
                                           n_steps=300, lr=1e-2, seed=0)
    pred_plain = np.asarray(obs_fn_plain(params_plain, valid_batch[0])[nac_key])
    mae_plain = sign_invariant_mae(pred_plain, valid_true_h)

    net_si = _build_net(prop_keys, F=32, n_layer=2, degrees=(1, 2), n_heads=1)
    obs_fn_si = jax.jit(jax.vmap(get_obs_and_force_fn(net_si), in_axes=(None, 0)))
    loss_fn_si = get_loss_fn(obs_fn=obs_fn_si, weights={pn.nac: 1.0}, prop_keys=prop_keys,
                             sign_invariant_targets=(pn.nac,))
    params_si, _, _ = _full_batch_train(net_si, loss_fn_si, train_batch, valid_batch,
                                        n_steps=300, lr=1e-2, seed=0)
    pred_si = np.asarray(obs_fn_si(params_si, valid_batch[0])[nac_key])
    mae_si = sign_invariant_mae(pred_si, valid_true_h)

    # empirically (fixed seeds -> deterministic): mae_plain ~0.069 (barely below the ~0.071
    # "predict zero" baseline -- plain MSE essentially fails to learn from scrambled signs),
    # mae_si ~0.063 (a real, if partial, recovery of the field). Both margins below have
    # comfortable headroom over the measured values.
    assert mae_si < mae_plain
    assert mae_si < 0.95 * zero_baseline
