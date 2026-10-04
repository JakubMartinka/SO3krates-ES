"""
End-to-end checks of the three PES variants -- single-state, multi-output (`nn.Energy(n_states=N)`)
and multi-state (`nn.StateEmbed` + `nn.MultiStateStackNet`) -- plus the `gap_weight` loss term and
the early-stopping return value of `Coach.run`.
"""
import pytest

from .test_data import load_data


def _gap_loss_setup(n_states):
    import jax.numpy as jnp
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    e_key = prop_keys[pn.energy]
    # The "params" are the predicted energies themselves, so the loss can be checked by hand.
    obs_fn = lambda params, inputs: {e_key: params}
    e_true = jnp.arange(4 * n_states, dtype=jnp.float32).reshape(4, n_states) ** 1.5
    e_pred = e_true + jnp.linspace(-0.3, 0.4, 4 * n_states).reshape(4, n_states)
    inputs = {prop_keys[pn.node_mask]: jnp.ones((4, 5), dtype=bool)}
    return obs_fn, e_pred, (inputs, {e_key: e_true}), e_true


def test_gap_loss_term():
    import numpy as np
    from mlff.training import get_loss_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    obs_fn, e_pred, batch, e_true = _gap_loss_setup(n_states=3)
    weights = {pn.energy: 2.}

    loss_plain, metrics_plain = get_loss_fn(obs_fn, weights, prop_keys)(e_pred, batch)
    loss_gap, metrics_gap = get_loss_fn(obs_fn, weights, prop_keys, gap_weight=0.5)(e_pred, batch)

    e_pred, e_true = np.asarray(e_pred), np.asarray(e_true)
    mse_energy = ((e_pred - e_true) ** 2).mean()
    mse_gap = ((np.diff(e_pred, axis=-1) - np.diff(e_true, axis=-1)) ** 2).mean()

    assert 'gap' not in metrics_plain
    np.testing.assert_allclose(loss_plain, 2. * mse_energy, rtol=1e-6)
    np.testing.assert_allclose(metrics_gap['gap'], mse_gap, rtol=1e-6)
    np.testing.assert_allclose(loss_gap, 2. * mse_energy + 0.5 * mse_gap, rtol=1e-6)


def test_gap_loss_term_is_noop_for_single_state():
    import numpy as np
    from mlff.training import get_loss_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    obs_fn, e_pred, batch, _ = _gap_loss_setup(n_states=1)
    weights = {pn.energy: 1.}

    loss_plain, _ = get_loss_fn(obs_fn, weights, prop_keys)(e_pred, batch)
    loss_gap, metrics_gap = get_loss_fn(obs_fn, weights, prop_keys, gap_weight=0.5)(e_pred, batch)

    assert 'gap' not in metrics_gap
    np.testing.assert_allclose(loss_gap, loss_plain)


def test_loss_rejects_shape_mismatch():
    # A single-state force prediction (B, n, 3) against a target that kept its state axis,
    # (B, 1, n, 3), would otherwise broadcast to (B, B, n, 3) and silently never be learned.
    import jax.numpy as jnp
    from mlff.training import get_loss_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    f_key = prop_keys[pn.force]
    obs_fn = lambda params, inputs: {f_key: params}
    inputs = {prop_keys[pn.node_mask]: jnp.ones((4, 5), dtype=bool)}
    loss_fn = get_loss_fn(obs_fn, {pn.force: 1.}, prop_keys)

    loss_fn(jnp.zeros((4, 5, 3)), (inputs, {f_key: jnp.ones((4, 5, 3))}))  # matching: fine
    with pytest.raises(ValueError, match='does not match'):
        loss_fn(jnp.zeros((4, 5, 3)), (inputs, {f_key: jnp.ones((4, 1, 5, 3))}))


def _build(variant, n_states, F=32):
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    embeddings = [nn.AtomTypeEmbed(num_embeddings=100, features=F, prop_keys=prop_keys)]
    if variant == 'multi_state':
        embeddings.append(nn.StateEmbed(features=F, prop_keys=prop_keys))
    n_heads = n_states if variant == 'multi_output' else 1
    net = nn.So3krates(F=F,
                       n_layer=2,
                       prop_keys=prop_keys,
                       embeddings=embeddings,
                       obs=[nn.Energy(prop_keys=prop_keys, n_states=n_heads)],
                       geometry_embed_kwargs={'degrees': [1, 2]},
                       so3krates_layer_kwargs={'n_heads': 1, 'degrees': [1, 2]})
    model = nn.MultiStateStackNet(net, n_states=n_states) if variant == 'multi_state' else net
    return net, model


@pytest.mark.parametrize('variant,n_states,stop_early', [('single_state', 1, True),
                                                         ('multi_output', 2, False),
                                                         ('multi_state', 2, False)])
def test_train_reload_predict(tmp_path, variant, n_states, stop_early):
    import os
    import numpy as np
    import jax
    import jax.numpy as jnp

    from mlff.io import bundle_dicts, save_dict, read_json, load_params_from_ckpt_dir
    from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
    from mlff.data import DataTuple, DataSet
    from mlff.nn import get_obs_and_force_fn, MultiStateStackNet
    from mlff.nn.stacknet import init_stack_net
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    ckpt_dir = str(tmp_path / 'module')
    e_key, f_key = prop_keys[pn.energy], prop_keys[pn.force]

    data = dict(load_data('ethanol.npz'))
    if n_states > 1:
        # A synthetic second state (shifted and slightly reshaped copy of the first) -- enough to
        # exercise the multi-state shapes, the per-state shift and the gap term without new data.
        e1 = np.asarray(data[e_key]).reshape(-1, 1)
        f1 = np.asarray(data[f_key])
        data[e_key] = np.concatenate([e1, 1.1 * e1 + 1.234], axis=-1)  # shape: (n_data, 2)
        data[f_key] = np.stack([f1, 1.1 * f1], axis=1)  # shape: (n_data, 2, n_atoms, 3)

    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=20, n_valid=4, n_test=None, r_cut=5, training=True, seed=0)
    data_set.shift_x_by_mean_x(x=pn.energy)
    d = data_set.get_data_split()

    net, model = _build(variant, n_states)
    obs_fn = jax.vmap(get_obs_and_force_fn(model), in_axes=(None, 0))

    learning_rate = 1e-3
    opt = Optimizer()
    tx = opt.get(learning_rate=learning_rate)
    coach = Coach(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask],
                  targets=[pn.energy, pn.force],
                  epochs=2,
                  training_batch_size=2,
                  validation_batch_size=2,
                  loss_weights={pn.energy: 0.01, pn.force: 0.99},
                  ckpt_dir=ckpt_dir,
                  net_seed=0,
                  training_seed=0,
                  # With a flat learning rate, a threshold equal to it trips at the first log step.
                  stop_lr_min=learning_rate if stop_early else None)
    loss_fn = get_loss_fn(obs_fn=obs_fn, weights=coach.loss_weights, prop_keys=prop_keys,
                          gap_weight=0.01 if n_states > 1 else None)

    data_tuple = DataTuple(inputs=coach.inputs, targets=coach.targets, prop_keys=prop_keys)
    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    example = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
    params = model.init(jax.random.PRNGKey(0), example)
    train_state, h_train_state = create_train_state(net, params, tx, polyak_step_size=None)

    h = bundle_dicts([net.__dict_repr__(), opt.__dict_repr__(), coach.__dict_repr__(),
                      data_set.__dict_repr__(), h_train_state])
    save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

    stopped_early = coach.run(train_state=train_state,
                              train_ds=train_ds,
                              valid_ds=valid_ds,
                              loss_fn=loss_fn,
                              eval_every_t=5,
                              log_every_t=5,
                              restart_by_nan=True,
                              use_wandb=False)
    assert stopped_early is stop_early

    # Reload straight away: after an early stop the checkpoint must already be flushed to disk.
    params = load_params_from_ckpt_dir(ckpt_dir)
    net_reloaded = init_stack_net(read_json(os.path.join(ckpt_dir, 'hyperparameters.json')))
    model_reloaded = (MultiStateStackNet(net_reloaded, n_states=n_states)
                      if variant == 'multi_state' else net_reloaded)

    # Batched and jitted, the way both training and the mlatom interface call it.
    batch = jax.tree_map(lambda x: jnp.array(x[:3, ...]), train_ds[0])
    out = jax.jit(jax.vmap(get_obs_and_force_fn(model_reloaded), in_axes=(None, 0)))(params, batch)
    out_ref = jax.jit(jax.vmap(get_obs_and_force_fn(model), in_axes=(None, 0)))(params, batch)
    out = {k: v[0] for k, v in out.items()}
    out_ref = {k: v[0] for k, v in out_ref.items()}
    n_atoms = example[prop_keys[pn.atomic_position]].shape[0]

    assert out[e_key].shape == (n_states,)
    assert out[f_key].shape == ((n_atoms, 3) if n_states == 1 else (n_states, n_atoms, 3))
    assert np.all(np.isfinite(out[e_key])) and np.all(np.isfinite(out[f_key]))
    np.testing.assert_allclose(out[e_key], out_ref[e_key], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[f_key], out_ref[f_key], rtol=1e-5, atol=1e-6)
    if n_states > 1:
        # The states must actually be distinguished, by their heads or by the state input.
        assert not np.allclose(out[e_key][0], out[e_key][1])
        assert not np.allclose(out[f_key][0], out[f_key][1])
