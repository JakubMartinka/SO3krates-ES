"""
Tests for feeding the gradient-difference vector into the *backbone* rather than only into the
coupling readout (`nn.GradDiffSPHCEmbed` + `nn.GradDiffEmbed`).

Why this exists. `nn.InterstateCoupling(use_grad_diff=True)` consumes `g` at the very end, as one
additive term `s2 * a(x_i, |g_i|) * g_i` on the predicted coupling, so `g` can only move the
prediction *along its own direction* -- and on this project's fulvene data the true coupling is
close to orthogonal to the gradient difference (median |cos| 0.08-0.17 across the three test sets,
against ~0.13 for random vectors in 3N = 36 dimensions). Nothing else in the network ever sees `g`:
only the scalar `|g_i|` enters that gate, and the gate multiplies only the `g` term. So the
measured "the oracle gradient-difference input does not help" result is a statement about that
injection, not about the information.

`GradDiffSPHCEmbed` adds the *direction* into chi's degree-1 block and `GradDiffEmbed` the
log-*magnitude* into the invariant features, which is what lets the layers build the invariants
`g_i . g_j`, `g_i . r_ij`, `|g_i|` and condition the attention, the pair coefficients and the
readout gates on them. The properties worth pinning down:

- `g` actually reaches quantities it could not reach before -- the invariant features `x`, and
  through them a readout that never touches `g` itself (`test_backbone_grad_diff_reaches_*`),
- the injected direction is written into chi in real-spherical-harmonic order, so the result is
  equivariant under a *Cartesian* rotation of both positions and `g`
  (`test_backbone_grad_diff_is_rotation_equivariant`) -- the same ordering trap that
  `InterstateCoupling` documents,
- translation invariance survives (`test_backbone_grad_diff_is_translation_invariant`),
- both modules round-trip through `__dict_repr__` so a checkpoint rebuilds them
  (`test_modules_round_trip_through_dict_repr`),
- a network built without them is bit-identical to one built before they existed
  (`test_absent_by_default`).
"""
import numpy as np
import pytest

from .test_data import load_data


def _prop_keys():
    from mlff.properties import md17_property_keys
    return dict(md17_property_keys)


def _build_net(prop_keys, backbone_grad_diff, readout_grad_diff=False, readout='pair',
               F=32, n_layer=2, degrees=(1, 2), n_heads=1):
    from mlff import nn

    obs = [nn.Energy(prop_keys=prop_keys, n_states=1),
           nn.InterstateCoupling(prop_keys=prop_keys, degrees=list(degrees), readout=readout,
                                 use_grad_diff=readout_grad_diff, enforce_sum_rule=True)]
    embeddings = [nn.AtomTypeEmbed(num_embeddings=100, features=F, prop_keys=prop_keys)]
    extra_geom = []
    if backbone_grad_diff:
        extra_geom.append(nn.GradDiffSPHCEmbed(prop_keys=prop_keys, degrees=list(degrees)))
        embeddings.append(nn.GradDiffEmbed(features=F, prop_keys=prop_keys))
    return nn.So3krates(F=F, n_layer=n_layer, prop_keys=prop_keys, obs=obs,
                        embeddings=embeddings,
                        extra_geometry_embeddings=extra_geom,
                        geometry_embed_kwargs={'degrees': list(degrees)},
                        so3krates_layer_kwargs={'n_heads': n_heads, 'degrees': list(degrees)})


def _batch(prop_keys, n_train=3, seed=0):
    """One real (ethanol) batch plus a synthetic gradient-difference input."""
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
    ds = jax.tree_util.tree_map(lambda x: jnp.array(x), data_tuple(d['train'])[0])
    # A gradient difference is a per-atom Cartesian vector with the same shape as the positions;
    # its magnitude spans a wide range in real data, so use a spread-out synthetic one here.
    R = np.asarray(ds[prop_keys[pn.atomic_position]])
    rng = np.random.default_rng(seed)
    g = rng.normal(size=R.shape) * rng.lognormal(0.0, 1.0, size=R.shape[:-1])[..., None]
    ds = dict(ds)
    ds[prop_keys[pn.grad_diff]] = jnp.array(g, dtype=jnp.float32)
    return ds


def _run(net, inputs, seed=0):
    import jax
    import jax.numpy as jnp
    from mlff.nn import get_observable_fn

    fn = jax.jit(jax.vmap(get_observable_fn(net), in_axes=(None, 0)))
    example = jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), inputs)
    params = net.init(jax.random.PRNGKey(seed), example)
    return params, fn, fn(params, inputs)


def _perturb_scales(params, factor=25.0):
    """Move the (deliberately small) grad-diff scales off their initialisation.

    Both modules start close to a network that ignores `g` -- `grad_diff_sphc_scale` at 0.1 and a
    freshly initialised `GradDiffEmbed` MLP -- which is right for training but would let these
    tests pass trivially on a near-zero effect. Scaling them up asks the question the trained
    model will face.
    """
    import jax

    def scale(path, leaf):
        name = '/'.join(str(k.key) for k in path if hasattr(k, 'key'))
        return leaf * factor if 'grad_diff' in name else leaf

    return jax.tree_util.tree_map_with_path(scale, params)


# --------------------------------------------------------------------------------------------
# registration / round-trip / backward compatibility
# --------------------------------------------------------------------------------------------

def test_modules_round_trip_through_dict_repr():
    from mlff import nn
    from mlff.nn.embed import get_embedding_module

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True)
    repr_ = net.__dict_repr__()['stack_net']

    geom = [x for x in repr_['geometry_embeddings'] if 'grad_diff_sphc_embed' in x]
    feat = [x for x in repr_['feature_embeddings'] if 'grad_diff_embed' in x]
    assert len(geom) == 1 and len(feat) == 1

    m1 = get_embedding_module('grad_diff_sphc_embed', geom[0]['grad_diff_sphc_embed'])
    m2 = get_embedding_module('grad_diff_embed', feat[0]['grad_diff_embed'])
    assert isinstance(m1, nn.GradDiffSPHCEmbed)
    assert isinstance(m2, nn.GradDiffEmbed)
    assert list(m1.degrees) == [1, 2]


def test_round_trips_through_init_stack_net():
    """The full checkpoint path: live model -> hyperparameters dict -> rebuilt model."""
    import jax
    import numpy as np
    from mlff.nn.stacknet import init_stack_net

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True)
    inputs = _batch(prop_keys)
    params, _, out = _run(net, inputs)

    rebuilt = init_stack_net(net.__dict_repr__())
    fn = jax.jit(jax.vmap(__import__('mlff').nn.get_observable_fn(rebuilt), in_axes=(None, 0)))
    out2 = fn(params, inputs)
    from mlff.properties import property_names as pn
    np.testing.assert_allclose(np.asarray(out[prop_keys[pn.nac]]),
                               np.asarray(out2[prop_keys[pn.nac]]), rtol=1e-6, atol=1e-6)


def test_absent_by_default():
    """A network built without the new modules must look exactly like one built before them."""
    prop_keys = _prop_keys()
    repr_ = _build_net(prop_keys, backbone_grad_diff=False).__dict_repr__()['stack_net']
    assert [list(x)[0] for x in repr_['geometry_embeddings']] == ['geometry_embed']
    assert [list(x)[0] for x in repr_['feature_embeddings']] == ['atom_type_embed']


# --------------------------------------------------------------------------------------------
# the point of the change: `g` reaches the representation
# --------------------------------------------------------------------------------------------

def test_backbone_grad_diff_reaches_a_readout_that_does_not_read_it():
    """With the backbone injection the coupling changes when `g` changes -- even though the
    readout here is `'pair'` with `use_grad_diff=False`, i.e. it never touches `g` itself."""
    import jax.numpy as jnp
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True, readout_grad_diff=False, readout='pair')
    inputs = _batch(prop_keys)
    params, fn, _ = _run(net, inputs)
    params = _perturb_scales(params)

    with_g = np.asarray(fn(params, inputs)[prop_keys[pn.nac]])
    zeroed = dict(inputs)
    zeroed[prop_keys[pn.grad_diff]] = jnp.zeros_like(inputs[prop_keys[pn.grad_diff]])
    without_g = np.asarray(fn(params, zeroed)[prop_keys[pn.nac]])

    rel = np.linalg.norm(with_g - without_g) / np.linalg.norm(with_g)
    assert rel > 1e-3, f'gradient difference had no effect on the prediction (rel={rel:.2e})'


def test_backbone_grad_diff_is_not_confined_to_the_g_direction():
    """The readout-only injection can only add a multiple of `g`. The backbone injection must not
    be limited that way -- that is the whole reason for it, since the true coupling is close to
    orthogonal to `g`."""
    import jax.numpy as jnp
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True, readout_grad_diff=False, readout='pair')
    inputs = _batch(prop_keys)
    params, fn, _ = _run(net, inputs)
    params = _perturb_scales(params)

    zeroed = dict(inputs)
    zeroed[prop_keys[pn.grad_diff]] = jnp.zeros_like(inputs[prop_keys[pn.grad_diff]])
    delta = (np.asarray(fn(params, inputs)[prop_keys[pn.nac]])
             - np.asarray(fn(params, zeroed)[prop_keys[pn.nac]]))          # (B,n,3)
    g = np.asarray(inputs[prop_keys[pn.grad_diff]])

    # component of the change orthogonal to g, per atom
    gn = np.linalg.norm(g, axis=-1, keepdims=True)
    g_hat = np.where(gn > 0, g / np.maximum(gn, 1e-30), 0.0)
    along = (delta * g_hat).sum(-1, keepdims=True) * g_hat
    perp = delta - along
    frac = np.linalg.norm(perp) / max(np.linalg.norm(delta), 1e-30)
    assert frac > 0.3, (f'the change induced by `g` is {1 - frac:.2%} collinear with `g`; the '
                        f'backbone path is supposed to lift that restriction')


def test_backbone_grad_diff_changes_invariant_features():
    """`GradDiffEmbed` must actually move `x`; a coupling head reads `x` through its gates."""
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True)
    inputs = _batch(prop_keys)
    params, _, _ = _run(net, inputs)
    params = _perturb_scales(params)

    # energy is a pure function of the invariant features, so a change in it proves `g` reached `x`
    fn = jax.jit(jax.vmap(nn.get_observable_fn(net), in_axes=(None, 0)))
    e1 = np.asarray(fn(params, inputs)[prop_keys[pn.energy]])
    zeroed = dict(inputs)
    zeroed[prop_keys[pn.grad_diff]] = jnp.zeros_like(inputs[prop_keys[pn.grad_diff]])
    e2 = np.asarray(fn(params, zeroed)[prop_keys[pn.energy]])
    assert np.abs(e1 - e2).max() > 1e-4


# --------------------------------------------------------------------------------------------
# symmetries -- easy to break when a raw Cartesian vector is written into an SPHC block
# --------------------------------------------------------------------------------------------

def test_backbone_grad_diff_is_rotation_equivariant():
    """Rotating positions *and* `g` must rotate the predicted coupling by the same matrix.

    This is the test that catches the real-spherical-harmonic ordering trap: chi's l=1 block is
    laid out as (y,z,x), so writing a Cartesian `g` into it without the permutation gives a
    conjugated rotation and fails here by tens of percent.
    """
    import jax.numpy as jnp
    from mlff.properties import property_names as pn
    from scipy.spatial.transform import Rotation

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True)
    inputs = _batch(prop_keys)
    params, fn, out = _run(net, inputs)
    params = _perturb_scales(params)

    M = Rotation.from_euler('xyz', [37.0, -18.0, 63.0], degrees=True).as_matrix()
    rot = dict(inputs)
    rot[prop_keys[pn.atomic_position]] = jnp.array(
        np.einsum('bai,ij->baj', np.asarray(inputs[prop_keys[pn.atomic_position]]), M.T))
    rot[prop_keys[pn.grad_diff]] = jnp.array(
        np.einsum('bai,ij->baj', np.asarray(inputs[prop_keys[pn.grad_diff]]), M.T))

    h = np.asarray(fn(params, inputs)[prop_keys[pn.nac]])
    h_rot = np.asarray(fn(params, rot)[prop_keys[pn.nac]])
    expected = np.einsum('bai,ij->baj', h, M.T)
    err = np.linalg.norm(h_rot - expected) / max(np.linalg.norm(expected), 1e-30)
    assert err < 1e-4, f'rotation equivariance broken, relative error {err:.3e}'


def test_backbone_grad_diff_is_translation_invariant():
    import jax.numpy as jnp
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    net = _build_net(prop_keys, backbone_grad_diff=True)
    inputs = _batch(prop_keys)
    params, fn, _ = _run(net, inputs)
    params = _perturb_scales(params)

    shifted = dict(inputs)
    shifted[prop_keys[pn.atomic_position]] = inputs[prop_keys[pn.atomic_position]] + jnp.array(
        [1.7, -0.4, 2.3])

    h = np.asarray(fn(params, inputs)[prop_keys[pn.nac]])
    h_shift = np.asarray(fn(params, shifted)[prop_keys[pn.nac]])
    err = np.linalg.norm(h - h_shift) / max(np.linalg.norm(h), 1e-30)
    assert err < 1e-5, f'translation invariance broken, relative error {err:.3e}'


def test_sphc_embed_is_the_identity_at_zero_scale():
    """`GradDiffSPHCEmbed` must add nothing at all when its learnable scale is zero -- the
    guarantee that the injection is a strict addition to chi and touches nothing else."""
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    mod = nn.GradDiffSPHCEmbed(prop_keys=prop_keys, degrees=[1, 2], scale_init=0.0)
    n, m_tot = 5, 3 + 5
    rng = np.random.default_rng(0)
    inputs = {'chi': jnp.array(rng.normal(size=(n, m_tot)), dtype=jnp.float32),
              'point_mask': jnp.ones((n,), dtype=jnp.float32),
              prop_keys[pn.grad_diff]: jnp.array(rng.normal(size=(n, 3)) * 10, dtype=jnp.float32)}
    params = mod.init(jax.random.PRNGKey(0), inputs)
    out = mod.apply(params, inputs)['chi']
    np.testing.assert_allclose(np.asarray(out), np.asarray(inputs['chi']), rtol=0, atol=1e-7)


def test_sphc_embed_writes_into_the_degree_one_block_only():
    """Only chi's l=1 block may change; the l=2 block is a different irrep and must be untouched."""
    import jax
    import jax.numpy as jnp
    from mlff import nn
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    mod = nn.GradDiffSPHCEmbed(prop_keys=prop_keys, degrees=[1, 2], scale_init=1.0)
    n = 5
    rng = np.random.default_rng(0)
    inputs = {'chi': jnp.array(rng.normal(size=(n, 3 + 5)), dtype=jnp.float32),
              'point_mask': jnp.ones((n,), dtype=jnp.float32),
              prop_keys[pn.grad_diff]: jnp.array(rng.normal(size=(n, 3)), dtype=jnp.float32)}
    params = mod.init(jax.random.PRNGKey(0), inputs)
    out = np.asarray(mod.apply(params, inputs)['chi'])
    chi = np.asarray(inputs['chi'])
    assert np.abs(out[:, :3] - chi[:, :3]).max() > 1e-3      # l=1 changed
    np.testing.assert_allclose(out[:, 3:], chi[:, 3:], rtol=0, atol=1e-7)   # l=2 untouched


def test_adding_the_modules_only_rescales_the_atom_type_embedding():
    """With both grad-diff paths zeroed the prediction is *nearly* -- not exactly -- the plain
    network's.

    `StackNet` sums its feature embeddings and divides by sqrt(n_embeds), so adding `GradDiffEmbed`
    rescales the atom-type embedding by 1/sqrt(2), and that feeds nonlinear layers. This is the
    same, documented consequence `StateEmbed` has; the test pins down that nothing *else* changes,
    by checking the residual is small rather than zero.
    """
    import jax
    import jax.numpy as jnp
    from mlff.properties import property_names as pn

    prop_keys = _prop_keys()
    plain = _build_net(prop_keys, backbone_grad_diff=False)
    withgd = _build_net(prop_keys, backbone_grad_diff=True)
    inputs = _batch(prop_keys)

    p_plain, fn_plain, _ = _run(plain, inputs)
    p_gd, fn_gd, _ = _run(withgd, inputs)

    merged = {'params': dict(p_gd['params'])}
    merged['params'].update(p_plain['params'])
    for k in list(merged['params']):
        if k.startswith('feature_embeddings_1') or k.startswith('geometry_embeddings_1'):
            merged['params'][k] = jax.tree_util.tree_map(jnp.zeros_like, merged['params'][k])

    h_plain = np.asarray(fn_plain(p_plain, inputs)[prop_keys[pn.nac]])
    h_gd = np.asarray(fn_gd(merged, inputs)[prop_keys[pn.nac]])
    c = (h_plain * h_gd).sum() / max(np.linalg.norm(h_plain) * np.linalg.norm(h_gd), 1e-30)
    assert c > 0.99, f'zeroed grad-diff path perturbs the network more than the 1/sqrt(2) feature '\
                     f'rescaling accounts for (cos={c:.4f})'
