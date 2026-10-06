"""
Energy shifts in `DataSet`, in particular the pooled least-squares shift (`shift_x_by_type_lse`) for
multi-state energies: one set of per-element atomic energies fitted over all states together and
subtracted from every state, so that it cancels in every gap.
"""
import numpy as np
import pytest

from mlff.data import DataSet
from mlff.properties import md17_property_keys as prop_keys
import mlff.properties.property_names as pn

E, Z = prop_keys[pn.energy], prop_keys[pn.atomic_type]


def _data_set(n_states, seed=0, n=(30, 10, 8), compositions=None):
    """
    A DataSet with train/valid/test splits set directly (no neighbour lists needed): structures of
    several compositions, padded with z = 0 to a common size, energies roughly extensive in the
    element counts plus a state-dependent part and noise.
    """
    rng = np.random.default_rng(seed)
    if compositions is None:
        compositions = [[6, 6, 1, 1, 1, 1], [6, 1, 1, 1, 1], [8, 1, 1], [6, 6, 8, 1, 1, 1, 1]]
    n_max = max(len(c) for c in compositions)
    self_energy = {1: -13.6, 6: -1030.0, 8: -2040.0}

    def split(n_split):
        z = np.zeros((n_split, n_max), dtype=int)
        e = np.zeros((n_split, n_states))
        for i in range(n_split):
            comp = compositions[rng.integers(len(compositions))]
            z[i, :len(comp)] = comp
            base = sum(self_energy[a] for a in comp) + rng.normal(scale=0.5)
            e[i] = base + np.arange(n_states) * rng.uniform(2.0, 5.0) + rng.normal(scale=0.1, size=n_states)
        return {E: e, Z: z}

    # The constructor only needs something of the right shape; the splits are what is shifted.
    first = split(1)
    ds = DataSet(prop_keys=prop_keys, data={E: first[E], Z: first[Z]})
    ds.data_split = {'train': split(n[0]), 'valid': split(n[1]), 'test': split(n[2])}
    return ds


def _per_structure(shift_table, z):
    table = np.asarray(shift_table, dtype=np.float64)
    table = table if table.ndim == 1 else table[:, 0]
    return np.take(table, z).sum(axis=-1)


@pytest.mark.parametrize('n_states', [2, 3])
def test_pooled_fit_equals_stacked_lstsq(n_states):
    ds = _data_set(n_states)
    e_train, z_train = ds.data_split['train'][E].copy(), ds.data_split['train'][Z].copy()
    ds.shift_x_by_type_lse(pn.energy)

    # Reference: every (structure, state) energy one equation in the element counts.
    elements = sorted(set(np.unique(z_train)) - {0})
    counts = np.array([[np.sum(zz == el) for el in elements] for zz in z_train], dtype=float)
    a = np.repeat(counts, n_states, axis=0)
    ref = np.linalg.lstsq(a, e_train.reshape(-1), rcond=None)[0]

    table = np.asarray(ds.scales[pn.energy]['per_atom_shift'])
    np.testing.assert_allclose(table[elements, 0], ref, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize('n_states', [2, 3])
def test_pooled_shift_keeps_every_gap_and_shifts_all_splits(n_states):
    ds = _data_set(n_states)
    before = {k: ds.data_split[k][E].copy() for k in ('train', 'valid', 'test')}
    ds.shift_x_by_type_lse(pn.energy)
    for k in ('train', 'valid', 'test'):
        after = ds.data_split[k][E]
        # Gaps unchanged, for every frame of every split ...
        np.testing.assert_allclose(np.diff(after, axis=-1), np.diff(before[k], axis=-1), atol=1e-9)
        # ... and every split, test included, actually shifted (by the per-structure total).
        shift = _per_structure(ds.scales[pn.energy]['per_atom_shift'], ds.data_split[k][Z])
        np.testing.assert_allclose(before[k] - after, np.repeat(shift[:, None], n_states, axis=1), atol=1e-9)


def test_pooled_shift_layout():
    n_states = 3
    ds = _data_set(n_states)
    ds.shift_x_by_type_lse(pn.energy)
    table = np.asarray(ds.scales[pn.energy]['per_atom_shift'])
    assert table.shape == (101, n_states)
    np.testing.assert_array_equal(table, np.repeat(table[:, :1], n_states, axis=1))  # identical columns
    np.testing.assert_array_equal(table[0], 0.)  # padding contributes nothing
    assert table[1, 0] != 0 and table[6, 0] != 0 and table[8, 0] != 0


def test_pooled_shift_single_composition_is_minimum_norm():
    ds = _data_set(2, compositions=[[6] * 6 + [1] * 6])  # one composition (C6H6)
    ds.shift_x_by_type_lse(pn.energy)
    table = np.asarray(ds.scales[pn.energy]['per_atom_shift'])
    # Only 6*e_C + 6*e_H is determined; the minimum-norm solution splits it equally.
    np.testing.assert_allclose(table[1, 0], table[6, 0], rtol=1e-10)
    # The pooled mean of the shifted training energies is ~0.
    assert abs(ds.data_split['train'][E].mean()) < 1e-8


def test_pooled_shift_rejects_missing_labels():
    ds = _data_set(2)
    ds.data_split['train'][E][3, 1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        ds.shift_x_by_type_lse(pn.energy)


def test_single_state_lse_and_mean_unchanged_on_train_and_valid():
    # Reference: the released implementations, re-stated here.
    ds = _data_set(1, seed=3)
    e = {k: ds.data_split[k][E].copy() for k in ('train', 'valid')}
    z = {k: ds.data_split[k][Z].copy() for k in ('train', 'valid')}
    ds.shift_x_by_type_lse(pn.energy)

    from mlff.data.preprocessing import get_per_atom_shift
    shifts, q_train = get_per_atom_shift(z=z['train'], q=e['train'].reshape(-1), pad_value=0)
    np.testing.assert_allclose(ds.data_split['train'][E].reshape(-1), q_train)
    np.testing.assert_allclose(ds.data_split['valid'][E].reshape(-1),
                               e['valid'].reshape(-1) - np.take(shifts, z['valid']).sum(axis=-1))
    assert ds.scales[pn.energy]['per_atom_shift'] == shifts.reshape(-1).tolist()

    ds = _data_set(1, seed=3)
    mean = ds.data_split['train'][E].reshape(-1).mean()
    before = {k: ds.data_split[k][E].copy() for k in ('train', 'valid')}
    ds.shift_x_by_mean_x(pn.energy)
    for k in ('train', 'valid'):
        np.testing.assert_allclose(ds.data_split[k][E], before[k] - mean)


def test_single_state_type_shifts_now_shift_the_test_split():
    for shift in ('lse', 'hand'):
        ds = _data_set(1, seed=4)
        before = ds.data_split['test'][E].copy()
        if shift == 'lse':
            ds.shift_x_by_type_lse(pn.energy)
        else:
            ds.shift_x_by_type_hand(pn.energy, shifts={1: -13.6, 6: -1030.0, 8: -2040.0})
        expected = before.reshape(-1) - _per_structure(ds.scales[pn.energy]['per_atom_shift'],
                                                       ds.data_split['test'][Z])
        np.testing.assert_allclose(ds.data_split['test'][E].reshape(-1), expected)


def test_hand_shift_applies_to_every_state():
    ds = _data_set(2, seed=5)
    before = ds.data_split['train'][E].copy()
    ds.shift_x_by_type_hand(pn.energy, shifts={1: -13.6, 6: -1030.0, 8: -2040.0})
    np.testing.assert_allclose(np.diff(ds.data_split['train'][E], axis=-1), np.diff(before, axis=-1), atol=1e-9)
    assert np.asarray(ds.scales[pn.energy]['per_atom_shift']).shape == (101, 2)
