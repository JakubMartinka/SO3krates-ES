from .test_data import load_data


def test_energy_multi_state_dict_repr():
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    n_states = 3
    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states)]
    net = nn.So3krates(n_layer=1, prop_keys=prop_keys, obs=obs)

    h = net.__dict_repr__()
    stack_net_obs = h['stack_net']['observables']
    assert stack_net_obs[0]['energy']['n_states'] == n_states


def test_energy_multi_state_forward_shapes():
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    n_states = 2
    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states)]
    net = nn.So3krates(F=36, n_layer=2, prop_keys=prop_keys, obs=obs)

    inputs = {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(0), shape=(7, 3)),
              prop_keys[pn.atomic_type]: jnp.ones(7),
              prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
              prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}

    params = net.init(jax.random.PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    energy_key = prop_keys[pn.energy]
    assert out[energy_key].shape == (n_states,)


def test_energy_multi_state_force_shapes():
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    n_states = 2
    n_atoms = 7
    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states)]
    net = nn.So3krates(F=36, n_layer=2, prop_keys=prop_keys, obs=obs)

    inputs = {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(0), shape=(n_atoms, 3)),
              prop_keys[pn.atomic_type]: jnp.ones(n_atoms),
              prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
              prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}

    params = net.init(jax.random.PRNGKey(0), inputs)

    obs_and_force_fn = get_obs_and_force_fn(net)
    out = obs_and_force_fn(params, inputs)

    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    assert out[energy_key].shape == (n_states,)
    assert out[force_key].shape == (n_states, n_atoms, 3)


def test_so3krates_multi_state_training():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import os
    import pathlib

    from mlff.io import create_directory, bundle_dicts, save_dict
    from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
    from mlff.data import DataTuple, DataSet

    from mlff.nn import get_obs_and_force_fn
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    import mlff.properties.property_names as pn

    n_states = 2

    data_path = 'test_data/ethanol.npz'
    save_path = pathlib.Path('_test_train_so3krates_multi_state').expanduser().absolute().resolve()
    ckpt_dir = os.path.join(save_path, 'module')
    ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

    data = dict(load_data('ethanol.npz'))

    # Synthesize a second, offset "excited state" column purely to exercise the multi-state
    # pipeline (DataSet shapes, per-state mean shift, loss masking, force autodiff) without
    # needing new fixture data. State 2 = state 1 energy + constant shift; forces are shared.
    e_key, f_key = prop_keys[pn.energy], prop_keys[pn.force]
    e1 = np.asarray(data[e_key]).reshape(-1, 1)
    f1 = np.asarray(data[f_key])
    data[e_key] = np.concatenate([e1, e1 + 1.234], axis=-1)  # shape: (n_data, 2)
    data[f_key] = np.stack([f1, f1], axis=1)  # shape: (n_data, 2, n_atoms, 3)

    r_cut = 5
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=50,
                          n_valid=10,
                          n_test=None,
                          r_cut=r_cut,
                          training=True,
                          seed=0)

    data_set.shift_x_by_mean_x(x=pn.energy)

    data_set.save_splits_to_file(ckpt_dir, 'splits.json')
    data_set.save_scales(ckpt_dir, 'scales.json')

    d = data_set.get_data_split()

    assert d['train'][e_key].shape == (50, n_states)
    assert d['train'][f_key].shape[:2] == (50, n_states)

    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states)]
    net = nn.So3krates(F=32,
                       n_layer=2,
                       prop_keys=prop_keys,
                       obs=obs,
                       geometry_embed_kwargs={'degrees': [1, 2]},
                       so3krates_layer_kwargs={'n_heads': 1,
                                               'degrees': [1, 2]})

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

    opt = Optimizer()
    tx = opt.get(learning_rate=1e-3)

    coach = Coach(inputs=[pn.atomic_position, pn.atomic_type, pn.idx_i, pn.idx_j, pn.node_mask],
                  targets=[pn.energy, pn.force],
                  epochs=2,
                  training_batch_size=2,
                  validation_batch_size=2,
                  loss_weights={pn.energy: 0.01, pn.force: 0.99},
                  ckpt_dir=ckpt_dir,
                  data_path=data_path,
                  net_seed=0,
                  training_seed=0)

    loss_fn = get_loss_fn(obs_fn=obs_fn,
                          weights=coach.loss_weights,
                          prop_keys=prop_keys)

    data_tuple = DataTuple(inputs=coach.inputs,
                           targets=coach.targets,
                           prop_keys=prop_keys)

    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    inputs = jax.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
    params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
    train_state, h_train_state = create_train_state(net,
                                                    params,
                                                    tx,
                                                    polyak_step_size=None)

    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = data_set.__dict_repr__()
    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
    save_dict(path=ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

    coach.run(train_state=train_state,
              train_ds=train_ds,
              valid_ds=valid_ds,
              loss_fn=loss_fn,
              eval_every_t=50,
              log_every_t=1,
              restart_by_nan=True,
              use_wandb=False)

    assert os.path.isfile(os.path.join(ckpt_dir, 'scales.json'))
    assert os.path.isfile(os.path.join(ckpt_dir, 'hyperparameters.json'))


def test_remove_dirs():
    try:
        import shutil
        shutil.rmtree('_test_train_so3krates_multi_state')
    except FileNotFoundError:
        pass
