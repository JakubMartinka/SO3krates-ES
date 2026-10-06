"""
`nn.StateInputEnergy`, the energy head of the *multi-state* variant: the raw state index appended
to every atom's descriptor (the final invariant features), read by one MLP shared by all states.
The So3krates layers run once; only the head runs per state.
"""
import numpy as np
import pytest


def _inputs(n_atoms=7, seed=0):
    import jax
    import jax.numpy as jnp
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    return {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(seed), shape=(n_atoms, 3)),
            prop_keys[pn.atomic_type]: jnp.array([6, 6, 1, 1, 8, 1, 1]),
            prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
            prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}


def _net(n_states, F=36, **energy_kwargs):
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    return nn.So3krates(F=F, n_layer=2, prop_keys=prop_keys,
                        obs=[nn.StateInputEnergy(prop_keys=prop_keys, n_states=n_states, **energy_kwargs)])


@pytest.mark.parametrize('n_states', [1, 2, 3])
def test_shapes(n_states):
    import jax
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net, inputs = _net(n_states), _inputs()
    params = net.init(jax.random.PRNGKey(0), inputs)
    out = jax.jit(get_obs_and_force_fn(net))(params, inputs)
    assert out[prop_keys[pn.energy]].shape == (n_states,)
    # Single-state forces keep the plain (n, 3) shape, as for `Energy`.
    assert out[prop_keys[pn.force]].shape == ((7, 3) if n_states == 1 else (n_states, 7, 3))


def test_per_atom_output_shape():
    import jax
    from mlff import nn
    from mlff.properties import md17_property_keys
    import mlff.properties.property_names as pn

    # The default key table has no atomic-energy entry; per-atom output needs one, as for `Energy`.
    prop_keys = {**md17_property_keys, pn.atomic_energy: 'E_atom'}
    net = nn.So3krates(F=36, n_layer=2, prop_keys=prop_keys,
                       obs=[nn.StateInputEnergy(prop_keys=prop_keys, n_states=3, output_convention='per_atom')])
    inputs = _inputs()
    params = net.init(jax.random.PRNGKey(0), inputs)
    assert net.apply(params, inputs)['E_atom'].shape == (7, 3)


def test_state_changes_energy_and_parameters_do_not_depend_on_n_states():
    import jax
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    inputs = _inputs()
    energies, n_params = {}, {}
    for n_states in (2, 3):
        net = _net(n_states)
        params = net.init(jax.random.PRNGKey(0), inputs)
        n_params[n_states] = sum(x.size for x in jax.tree_util.tree_leaves(params['params']))
        energies[n_states] = np.asarray(net.apply(params, inputs)[prop_keys[pn.energy]])

    assert not np.isclose(energies[2][0], energies[2][1])
    assert n_params[2] == n_params[3]
    # One set of weights for every state: the first states do not depend on how many there are.
    np.testing.assert_allclose(energies[3][:2], energies[2], rtol=1e-6)


def test_states_differ_only_through_the_state_input():
    # Same network, same parameters: setting the state column of the head input to a constant
    # must make every state identical, i.e. nothing else in the head depends on the state.
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net, inputs = _net(3), _inputs()
    params = net.init(jax.random.PRNGKey(0), inputs)
    head_params = params['params']['observables_0']
    first = head_params['MLP_0']['layers_0']['kernel']  # shape: (F + 1, F); last row reads the state
    zeroed = jax.tree_util.tree_map(lambda x: x, params)
    zeroed['params']['observables_0']['MLP_0']['layers_0']['kernel'] = first.at[-1].set(0.)
    e = net.apply(zeroed, inputs)[prop_keys[pn.energy]]
    assert jnp.allclose(e, e[0])


def test_rotation_invariance_and_equivariance():
    import jax
    import jax.numpy as jnp
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net, inputs = _net(2), _inputs()
    params = net.init(jax.random.PRNGKey(0), inputs)
    fn = jax.jit(get_obs_and_force_fn(net))

    a, b, c = 0.3, -1.1, 2.0  # an arbitrary rotation
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    rot = jnp.asarray(rz @ ry @ rx, dtype=jnp.float32)

    out = fn(params, inputs)
    r_key = prop_keys[pn.atomic_position]
    out_rot = fn(params, {**inputs, r_key: inputs[r_key] @ rot.T})
    np.testing.assert_allclose(out_rot[prop_keys[pn.energy]], out[prop_keys[pn.energy]], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_rot[prop_keys[pn.force]], out[prop_keys[pn.force]] @ rot.T, rtol=1e-4, atol=1e-5)


def test_dict_repr_round_trip():
    import jax
    from mlff.nn.stacknet import init_stack_net
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net, inputs = _net(3), _inputs()
    h = net.__dict_repr__()
    obs = h['stack_net']['observables'][0]
    assert list(obs) == ['state_input_energy'] and obs['state_input_energy']['n_states'] == 3

    rebuilt = init_stack_net(h)
    params = net.init(jax.random.PRNGKey(0), inputs)
    np.testing.assert_allclose(rebuilt.apply(params, inputs)[prop_keys[pn.energy]],
                               net.apply(params, inputs)[prop_keys[pn.energy]], rtol=1e-6)
