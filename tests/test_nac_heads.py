"""
Tests for the alternative nonadiabatic-coupling readouts added alongside the original
`nn.InterstateCoupling(readout='chi')` head, and for the per-example loss weighting that goes
with them.

Why these exist. The original head is `h_i = s * a(x_i) * chi_i[l=1]`, and `chi` carries exactly
one channel per angular-momentum degree -- so the *direction* of the predicted vector at atom `i`
is pinned by the geometry and only a signed magnitude is learnable. That is one degree of freedom
per atom for a three-degree-of-freedom target, and it also leaves the translational sum rule
`sum_i h_i = 0` (which a real coupling obeys, and which this project's reference data satisfies
to ~0.4% median) unenforced -- a trained `'chi'` head was measured violating it by ~11%. The two
new readouts remove both problems in different ways, so the properties worth pinning down are:

- the pair readout spans all of R^3 rather than a fixed line (`test_pair_readout_is_not_collinear
  _with_chi_l1`, `test_pair_readout_spans_three_dimensions`),
- it satisfies the sum rule *exactly*, by construction rather than by fitting
  (`test_pair_readout_satisfies_translational_sum_rule`),
- `enforce_sum_rule` retrofits the same guarantee onto the `'chi'` head
  (`test_enforce_sum_rule_projects_out_net_translation`),
- all readouts stay rotation-equivariant and translation-invariant, which is easy to break when
  a head starts mixing raw Cartesian pair vectors (`test_*_is_rotation_equivariant`,
  `test_*_is_translation_invariant`),
- the defaults reproduce old checkpoints bit-for-bit (`test_defaults_match_original_head`), since
  the new fields are read back from `hyperparameters.json` files that predate them,
- the weighted loss actually re-weights per example rather than silently averaging
  (`test_sample_weighted_loss_*`).
"""
import numpy as np
import pytest

from .test_data import load_data


def _nac_prop_keys():
    from mlff.properties import md17_property_keys
    return dict(md17_property_keys)


def _build_net(prop_keys, readout='chi', enforce_sum_rule=False, F=32, n_layer=2,
               degrees=(1, 2), n_heads=1, n_states=1, potential_head=False):
    from mlff import nn

    obs = [nn.Energy(prop_keys=prop_keys, n_states=n_states)]
    if potential_head:
        obs.append(nn.InterstateCouplingPotential(prop_keys=prop_keys))
    else:
        obs.append(nn.InterstateCoupling(prop_keys=prop_keys, degrees=list(degrees),
                                         readout=readout, enforce_sum_rule=enforce_sum_rule))
    return nn.So3krates(F=F, n_layer=n_layer, prop_keys=prop_keys, obs=obs,
                        geometry_embed_kwargs={'degrees': list(degrees)},
                        so3krates_layer_kwargs={'n_heads': n_heads, 'degrees': list(degrees)})


def _batch(prop_keys, n_train=3):
    """One real (ethanol) batch, plus the initialised params, for a given net."""
    import jax
    import jax.numpy as jnp
    from mlff.properties import property_names as pn
    from mlff.data import DataSet, DataTuple

    data = dict(load_data('ethanol.npz'))
    data_set = DataSet(data=data, prop_keys=prop_keys)
    data_set.random_split(n_train=n_train, n_valid=1, n_test=None, mic=False, r_cut=5,
                          training=True, seed=0)
    d = data_set.get_data_split()
    data_tuple = DataTuple(inputs=[pn.atomic_position, pn.atomic_type, pn.node_mask,
                                   pn.idx_i, pn.idx_j],
                           targets=[pn.energy, pn.force],
                           prop_keys=prop_keys)
    ds = data_tuple(d['train'])
    return jax.tree_util.tree_map(lambda x: jnp.array(x), ds[0])


def _nac_of(net, inputs, seed=0, nac_from_potential=False):
    import jax
    import jax.numpy as jnp
    from mlff.nn import get_obs_and_force_fn
    from mlff.properties import property_names as pn

    obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(net, nac_from_potential=nac_from_potential),
                              in_axes=(None, 0)))
    example = jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), inputs)
    params = net.init(jax.random.PRNGKey(seed), example)
    return params, obs_fn, obs_fn(params, inputs)[_nac_prop_keys()[pn.nac]]


# --------------------------------------------------------------------------------------------
# registration / round-trip
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize('readout', ['chi', 'pair', 'chi+pair'])
def test_readout_round_trips_through_dict_repr(readout):
    from mlff import nn
    from mlff.nn.observable import get_observable_module

    prop_keys = _nac_prop_keys()
    net = _build_net(prop_keys, readout=readout, enforce_sum_rule=True)
    nac_repr = [o for o in net.__dict_repr__()['stack_net']['observables'] if 'nac' in o][0]['nac']

    assert nac_repr['readout'] == readout
    assert nac_repr['enforce_sum_rule'] is True

    mod = get_observable_module('nac', nac_repr)
    assert isinstance(mod, nn.InterstateCoupling)
    assert mod.readout == readout and mod.enforce_sum_rule is True


def test_coupling_potential_round_trips_through_dict_repr():
    from mlff import nn
    from mlff.nn.observable import get_observable_module

    prop_keys = _nac_prop_keys()
    net = _build_net(prop_keys, potential_head=True)
    names = [list(o.keys())[0] for o in net.__dict_repr__()['stack_net']['observables']]
    assert 'nac_potential' in names

    repr_ = net.__dict_repr__()['stack_net']['observables'][names.index('nac_potential')]
    mod = get_observable_module('nac_potential', repr_['nac_potential'])
    assert isinstance(mod, nn.InterstateCouplingPotential)


def test_defaults_match_original_head():
    """A hyperparameters.json written before `readout`/`enforce_sum_rule` existed must rebuild
    the original head exactly, or every previously trained checkpoint silently changes."""
    from mlff.nn.observable import get_observable_module

    prop_keys = _nac_prop_keys()
    legacy = {'degrees': [1, 2], 'output_scale': 1.0, 'use_grad_diff': False,
              'prop_keys': prop_keys}
    mod = get_observable_module('nac', legacy)
    assert mod.readout == 'chi'
    assert mod.enforce_sum_rule is False


def test_invalid_readout_rejected():
    import jax
    import jax.numpy as jnp
    from mlff import nn

    mod = nn.InterstateCoupling(prop_keys=_nac_prop_keys(), degrees=[1, 2], readout='bogus')
    with pytest.raises(ValueError):
        mod.init(jax.random.PRNGKey(0),
                 {'x': jnp.ones((3, 4)), 'chi': jnp.ones((3, 8)), 'point_mask': jnp.ones(3)})


def test_pair_readout_does_not_require_degree_one():
    """Unlike the chi readout, the pair readout never touches chi's l=1 block."""
    import jax
    from mlff import nn

    prop_keys = _nac_prop_keys()
    net = _build_net(prop_keys, readout='pair', degrees=(2, 3))
    inputs = _batch(prop_keys)
    params, _, nac = _nac_of(net, inputs)
    assert np.isfinite(np.asarray(nac)).all()
    assert float(np.abs(np.asarray(nac)).max()) > 0


# --------------------------------------------------------------------------------------------
# expressiveness: the whole point of the pair readout
# --------------------------------------------------------------------------------------------

def test_pair_readout_is_not_collinear_with_chi_l1():
    """The 'chi' head's output is *exactly* parallel to chi[l=1] (that is its known 1-DOF
    limitation); the pair head must not be."""
    import jax
    import jax.numpy as jnp

    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)

    def cos_with_chi_l1(readout):
        net = _build_net(prop_keys, readout=readout)
        params, _, nac = _nac_of(net, inputs)
        _, state = jax.vmap(lambda p, x: net.apply(p, x, mutable=['record']),
                            in_axes=(None, 0))(params, inputs)
        # last layer's post-update chi, whatever the submodule naming is
        leaves = [v for p, v in jax.tree_util.tree_flatten_with_path(state['record'])[0]
                  if 'chi_out' in jax.tree_util.keystr(p)]
        chi = np.asarray(leaves[-1]).reshape(nac.shape[0], nac.shape[1], -1)
        chi_l1 = chi[:, :, 0:3][:, :, [2, 0, 1]]  # degrees=(1,2) -> l=1 block is first
        nac = np.asarray(nac)
        denom = np.linalg.norm(nac, axis=-1) * np.linalg.norm(chi_l1, axis=-1)
        return np.abs(np.sum(nac * chi_l1, -1))[denom > 0] / denom[denom > 0]

    assert np.allclose(cos_with_chi_l1('chi'), 1.0, atol=1e-4)
    assert np.median(cos_with_chi_l1('pair')) < 0.99


def test_pair_readout_spans_three_dimensions():
    """Scanning the pair filter's parameters must move the output off any fixed line: the head
    has to be able to point anywhere in R^3, not just rescale a geometry-fixed direction."""
    import jax
    import jax.numpy as jnp

    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    net = _build_net(prop_keys, readout='pair')
    params, obs_fn, base = _nac_of(net, inputs)
    from mlff.properties import property_names as pn
    nac_key = prop_keys[pn.nac]

    # collect the per-atom output for several random parameter draws; if the head were confined
    # to a fixed per-atom direction, every draw would be a multiple of the same vector and the
    # stacked matrix would have rank 1.
    outs = []
    for seed in range(1, 7):
        p = net.init(jax.random.PRNGKey(seed),
                     jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), inputs))
        outs.append(np.asarray(obs_fn(p, inputs)[nac_key]))
    stacked = np.stack(outs, axis=-2)  # (B, n_atoms, n_draws, 3)
    ranks = np.linalg.matrix_rank(stacked, tol=1e-6)
    assert np.median(ranks) == 3


# --------------------------------------------------------------------------------------------
# physical constraints
# --------------------------------------------------------------------------------------------

def _sum_rule_violation(nac, node_mask):
    nac = np.asarray(nac)
    net_sum = np.linalg.norm(nac.sum(axis=1), axis=-1)
    scale = np.linalg.norm(nac.reshape(nac.shape[0], -1), axis=-1)
    return net_sum / np.maximum(scale, 1e-30)


def test_pair_readout_satisfies_translational_sum_rule():
    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    net = _build_net(prop_keys, readout='pair')
    _, _, nac = _nac_of(net, inputs)
    assert float(np.abs(np.asarray(nac)).max()) > 0
    assert _sum_rule_violation(nac, inputs['node_mask']).max() < 1e-5


def test_chi_readout_violates_sum_rule_but_enforce_fixes_it():
    """Guards the motivation for `enforce_sum_rule`: without it the chi head has no reason to
    conserve momentum, and measurably does not."""
    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)

    _, _, plain = _nac_of(_build_net(prop_keys, readout='chi'), inputs)
    _, _, fixed = _nac_of(_build_net(prop_keys, readout='chi', enforce_sum_rule=True), inputs)

    assert _sum_rule_violation(plain, inputs['node_mask']).max() > 1e-3
    assert _sum_rule_violation(fixed, inputs['node_mask']).max() < 1e-5


def test_enforce_sum_rule_projects_out_net_translation():
    """`enforce_sum_rule` must subtract exactly the mean, not distort the rest of the field."""
    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    _, _, plain = _nac_of(_build_net(prop_keys, readout='chi'), inputs, seed=3)
    _, _, fixed = _nac_of(_build_net(prop_keys, readout='chi', enforce_sum_rule=True),
                          inputs, seed=3)
    plain, fixed = np.asarray(plain), np.asarray(fixed)
    assert np.allclose(fixed, plain - plain.mean(axis=1, keepdims=True), atol=1e-5)


def test_coupling_potential_head_satisfies_sum_rule():
    """A gradient of a scalar that depends on positions only through relative displacements
    conserves momentum automatically -- the SchNarc-style head gets this for free."""
    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    net = _build_net(prop_keys, potential_head=True)
    _, _, nac = _nac_of(net, inputs, nac_from_potential=True)
    assert float(np.abs(np.asarray(nac)).max()) > 0
    assert _sum_rule_violation(nac, inputs['node_mask']).max() < 1e-4


@pytest.mark.parametrize('readout,potential', [('pair', False), ('chi+pair', False),
                                               (None, True)])
def test_readout_is_rotation_equivariant(readout, potential):
    import jax.numpy as jnp
    from mlff.geometric import get_rotation_matrix, apply_rotation

    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    net = _build_net(prop_keys, readout=readout or 'chi', potential_head=potential)
    params, obs_fn, base = _nac_of(net, inputs, nac_from_potential=potential)
    from mlff.properties import property_names as pn
    nac_key = prop_keys[pn.nac]
    assert float(jnp.abs(base).max()) > 0

    rng = np.random.RandomState(0)
    for _ in range(3):
        M_rot = get_rotation_matrix(euler_axes='xyz', angles=rng.rand(3) * 360, degrees=True)
        rotated = {k: v for k, v in inputs.items()}
        rotated['R'] = apply_rotation(inputs['R'], m_rot=M_rot)
        out = obs_fn(params, rotated)[nac_key]
        assert not np.isclose(np.asarray(out).reshape(-1), np.asarray(base).reshape(-1)).all()
        assert np.allclose(np.asarray(apply_rotation(base, M_rot)).reshape(-1),
                           np.asarray(out).reshape(-1), atol=1e-4)


@pytest.mark.parametrize('readout,potential', [('chi', False), ('pair', False),
                                               ('chi+pair', False), (None, True)])
def test_readout_is_translation_invariant(readout, potential):
    """Every readout must be built from relative displacements only. (SpaiNN's PaiNN NAC head
    adds the atom's *absolute* position into the output vector -- a heuristic that would fail
    exactly this check; see PROJECT_NOTES.md.)"""
    prop_keys = _nac_prop_keys()
    inputs = _batch(prop_keys)
    net = _build_net(prop_keys, readout=readout or 'chi', potential_head=potential)
    params, obs_fn, base = _nac_of(net, inputs, nac_from_potential=potential)
    from mlff.properties import property_names as pn
    nac_key = prop_keys[pn.nac]

    shifted = {k: v for k, v in inputs.items()}
    shifted['R'] = inputs['R'] + np.array([3.7, -1.2, 0.9], dtype=np.float32)
    out = obs_fn(params, shifted)[nac_key]
    assert np.allclose(np.asarray(base), np.asarray(out), atol=1e-4)


# --------------------------------------------------------------------------------------------
# per-example loss weighting
# --------------------------------------------------------------------------------------------

def _weighted_loss_inputs(weights):
    import jax.numpy as jnp
    from mlff.properties import property_names as pn

    prop_keys = _nac_prop_keys()
    B, n_atoms = len(weights), 4
    inputs = {prop_keys[pn.node_mask]: jnp.ones((B, n_atoms), dtype=bool),
              prop_keys[pn.nac_sample_weight]: jnp.array(weights, dtype=jnp.float32)[:, None]}
    target = jnp.zeros((B, n_atoms, 3))
    # per-example error grows with the example index, so a weighting change is visible
    pred = jnp.arange(B, dtype=jnp.float32)[:, None, None] * jnp.ones((1, n_atoms, 3))
    return prop_keys, inputs, {prop_keys[pn.nac]: target}, {prop_keys[pn.nac]: pred}


def _run_loss(weights, sample_weighted, sign_invariant=False):
    from mlff.training import get_loss_fn
    from mlff.properties import property_names as pn

    prop_keys, inputs, targets, outputs = _weighted_loss_inputs(weights)
    loss_fn = get_loss_fn(obs_fn=lambda p, x: outputs,
                          weights={pn.nac: 1.0},
                          prop_keys=prop_keys,
                          sign_invariant_targets=(pn.nac,) if sign_invariant else (),
                          sample_weighted_targets=(pn.nac,) if sample_weighted else (),
                          sample_weight_key=pn.nac_sample_weight if sample_weighted else None)
    loss, _ = loss_fn({}, (inputs, targets))
    return float(loss)


def test_sample_weighted_loss_matches_plain_mean_for_uniform_weights():
    """Uniform weights must reproduce the unweighted loss exactly -- otherwise turning the
    feature on would silently rescale the loss (and the effective learning rate) as a side
    effect of enabling it."""
    assert _run_loss([1.0] * 4, sample_weighted=True) == pytest.approx(
        _run_loss([1.0] * 4, sample_weighted=False), rel=1e-6)


def test_sample_weighted_loss_is_normalized_by_its_own_weight_sum():
    """Scaling every weight by a constant must not change the loss, so the global normalization
    of the weights cannot leak into the learning rate."""
    assert _run_loss([1.0, 2.0, 3.0, 4.0], sample_weighted=True) == pytest.approx(
        _run_loss([10.0, 20.0, 30.0, 40.0], sample_weighted=True), rel=1e-6)


def test_sample_weighted_loss_actually_reweights():
    """Up-weighting the high-error examples must raise the loss, down-weighting must lower it --
    the whole point being to move error budget towards the near-conical-intersection points."""
    uniform = _run_loss([1.0] * 4, sample_weighted=True)
    emphasize_late = _run_loss([0.1, 0.1, 1.0, 10.0], sample_weighted=True)
    emphasize_early = _run_loss([10.0, 1.0, 0.1, 0.1], sample_weighted=True)
    assert emphasize_late > uniform > emphasize_early


def test_sample_weighting_composes_with_sign_invariance():
    """The two features touch the same per-example code path; enabling both must still weight."""
    uniform = _run_loss([1.0] * 4, sample_weighted=True, sign_invariant=True)
    weighted = _run_loss([0.1, 0.1, 1.0, 10.0], sample_weighted=True, sign_invariant=True)
    assert weighted > uniform
    # sign invariance is still active: the target is zero here, so +/- agree and the value must
    # match the non-sign-invariant weighted loss
    assert weighted == pytest.approx(_run_loss([0.1, 0.1, 1.0, 10.0], sample_weighted=True),
                                     rel=1e-6)


def test_sample_weighted_targets_requires_a_key():
    from mlff.training import get_loss_fn
    from mlff.properties import property_names as pn

    with pytest.raises(ValueError):
        get_loss_fn(obs_fn=lambda p, x: {}, weights={pn.nac: 1.0},
                    prop_keys=_nac_prop_keys(), sample_weighted_targets=(pn.nac,))
