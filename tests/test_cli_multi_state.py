"""
The command-line path for multi-state data: `train_so3krates --n_states 2 --shift_by lse` trains
with the pooled per-element energy shift, and `evaluate` reports finite errors for the result.
"""
import json
import os
import subprocess
import sys

import numpy as np

from .test_data import load_data


def _console_script(name):
    # The console scripts are installed next to the interpreter running the tests.
    return os.path.join(os.path.dirname(sys.executable), name)


def test_cli_multi_state_lse_trains_and_evaluates(tmp_path):
    data = dict(load_data('ethanol.npz'))
    e1 = np.asarray(data['E']).reshape(-1, 1)[:200]
    f1 = np.asarray(data['F'])[:200]
    # A synthetic second state, as in test_pes_variants.py: enough to exercise the multi-state path.
    np.savez(tmp_path / 'two_states.npz', R=np.asarray(data['R'])[:200], z=data['z'],
             E=np.concatenate([e1, e1 + 2.5], axis=-1), F=np.stack([f1, 1.1 * f1], axis=1))

    env = {**os.environ, 'WANDB_MODE': 'disabled'}
    ckpt_dir = tmp_path / 'module'
    train = subprocess.run([_console_script('train_so3krates'),
                            '--data_file', str(tmp_path / 'two_states.npz'),
                            '--n_train', '60', '--n_valid', '20', '--n_test', '40',
                            '--n_states', '2', '--shift_by', 'lse', '--epochs', '2',
                            '--L', '1', '--F', '16', '--degrees', '1', '2',
                            '--ckpt_dir', str(ckpt_dir)],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1200)
    assert train.returncode == 0, train.stderr[-3000:]

    # The pooled shift: one per-element fit, identical for both states, zero for padding.
    table = np.asarray(json.load(open(ckpt_dir / 'scales.json'))['energy']['per_atom_shift'])
    assert table.shape == (101, 2)
    np.testing.assert_array_equal(table[:, 0], table[:, 1])
    np.testing.assert_array_equal(table[0], 0.)

    evaluate = subprocess.run([_console_script('evaluate'), '--n_test', '40'],
                              cwd=ckpt_dir, env=env, capture_output=True, text=True, timeout=1200)
    assert evaluate.returncode == 0, evaluate.stderr[-3000:]
    metrics = json.load(open(ckpt_dir / 'metrics_on_test.json'))
    for kind in ('mae', 'rmse'):
        for prop in ('E', 'F'):
            assert np.isfinite(metrics[kind][prop]), (kind, prop, metrics)
