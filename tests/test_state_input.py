"""
The *multi-state* variant: one weight-shared network with the state index as an input
(`nn.StateEmbed`, `nn.MultiStateStackNet`), as opposed to the *multi-output* variant tested in
`test_multi_state.py`, where one pass emits every state from its own head.
"""


def _inputs(n_atoms=7, seed=0):
    import jax
    import jax.numpy as jnp
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    return {prop_keys[pn.atomic_position]: jax.random.normal(jax.random.PRNGKey(seed),
                                                             shape=(n_atoms, 3)),
            prop_keys[pn.atomic_type]: jnp.ones(n_atoms),
            prop_keys[pn.idx_i]: jnp.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6]),
            prop_keys[pn.idx_j]: jnp.array([1, 6, 0, 6, 4, 5, 6, 3, 3, 0, 2, 3])}


def _state_conditioned_net(F=36, n_layer=2):
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys

    embeddings = [nn.AtomTypeEmbed(num_embeddings=100, features=F, prop_keys=prop_keys),
                  nn.StateEmbed(features=F, prop_keys=prop_keys)]
    obs = [nn.Energy(prop_keys=prop_keys, n_states=1)]
    return nn.So3krates(F=F, n_layer=n_layer, prop_keys=prop_keys, embeddings=embeddings, obs=obs)


def test_state_embed_dict_repr_round_trip():
    from mlff.nn.stacknet import init_stack_net

    net = _state_conditioned_net()
    h = net.__dict_repr__()
    embed_names = [list(x.keys())[0] for x in h['stack_net']['feature_embeddings']]
    assert embed_names == ['atom_type_embed', 'state_embed']

    # The checkpoint of a multi-state model is a plain StackNet: it must rebuild from its own
    # hyperparameter dict, since that is how the interface reloads it.
    rebuilt = init_stack_net(h)
    assert [list(x.keys())[0] for x in rebuilt.__dict_repr__()['stack_net']['feature_embeddings']] \
        == embed_names


def test_state_input_changes_the_energy():
    import jax
    import jax.numpy as jnp
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net = _state_conditioned_net()
    inputs = _inputs()
    state_key = prop_keys[pn.state]
    energy_key = prop_keys[pn.energy]

    params = net.init(jax.random.PRNGKey(0), {**inputs, state_key: jnp.array([0.])})
    e0 = net.apply(params, {**inputs, state_key: jnp.array([0.])})[energy_key]
    e1 = net.apply(params, {**inputs, state_key: jnp.array([1.])})[energy_key]

    assert e0.shape == (1,)
    assert not jnp.allclose(e0, e1)

    # Zeroing the single learned state direction must collapse the states exactly: that parameter
    # is the only route by which the state index reaches the network.
    zeroed = jax.tree_util.tree_map(lambda x: x, params).unfreeze() \
        if hasattr(params, 'unfreeze') else jax.tree_util.tree_map(lambda x: x, params)
    zeroed['params']['feature_embeddings_1']['state_direction'] = jnp.zeros_like(
        zeroed['params']['feature_embeddings_1']['state_direction'])
    assert jnp.allclose(net.apply(zeroed, {**inputs, state_key: jnp.array([0.])})[energy_key],
                        net.apply(zeroed, {**inputs, state_key: jnp.array([1.])})[energy_key])


def test_multi_state_stack_net_shapes_and_values():
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    n_states = 3
    net = _state_conditioned_net()
    model = nn.MultiStateStackNet(net, n_states=n_states)
    inputs = _inputs()
    energy_key = prop_keys[pn.energy]
    force_key = prop_keys[pn.force]
    state_key = prop_keys[pn.state]

    params = model.init(jax.random.PRNGKey(0), inputs)
    out = model.apply(params, inputs)

    # Same output shape as the multi-output variant, which is what lets the loss, the energy-shift
    # bookkeeping and predict() stay variant-agnostic.
    assert out[energy_key].shape == (n_states,)

    # ... and the stacked values are exactly the per-state single passes.
    for s in range(n_states):
        single = net.apply(params, {**inputs, state_key: jnp.array([float(s)])})[energy_key]
        assert jnp.allclose(out[energy_key][s], single[0], atol=1e-6)

    # One jacrev still gives per-state forces.
    forces = get_obs_and_force_fn(model)(params, inputs)[force_key]
    assert forces.shape == (n_states, 7, 3)


def test_multi_state_stack_net_single_state_shapes():
    import jax
    from mlff import nn
    from mlff.properties import md17_property_keys as prop_keys
    import mlff.properties.property_names as pn

    net = _state_conditioned_net()
    model = nn.MultiStateStackNet(net, n_states=1)
    inputs = _inputs()
    params = model.init(jax.random.PRNGKey(0), inputs)
    assert model.apply(params, inputs)[prop_keys[pn.energy]].shape == (1,)
