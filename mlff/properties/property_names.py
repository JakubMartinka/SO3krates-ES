energy = 'energy'
force = 'force'
nac = 'nac'  # scaled interstate coupling h = NAC * (E_1 - E_0), shape (n_atoms, 3); see nn.InterstateCoupling
grad_diff = 'grad_diff'  # gradient-difference vector dE_1/dR - dE_0/dR, shape (n_atoms, 3); optional
                          # extra InterstateCoupling input, see InterstateCoupling.use_grad_diff
nac_potential = 'nac_potential'  # invariant scalar whose position-gradient is the coupling, shape (1);
                                  # see nn.InterstateCouplingPotential
nac_sample_weight = 'nac_sample_weight'  # per-geometry weight for the `nac` loss term, shape (1);
                                          # see training/loss.py::get_loss_fn's `sample_weight_key`
hirshfeld_volume = 'hirshfeld_volume'
hirshfeld_volume_ratio = 'hirshfeld_volume_ratio'
partial_charge = 'partial_charge'
total_dipole_moment = 'total_dipole_moment'
total_quadrupole_moment = 'total_quadrupole_moment'
stress = 'stress'
atomic_energy = 'atomic_energy'

atomic_position = 'atomic_position'
atomic_type = 'atomic_type'

total_charge = 'total_charge'
total_spin = 'total_spin'

idx_i = 'idx_i'
idx_j = 'idx_j'

node_mask = 'node_mask'

unit_cell = 'unit_cell'
cell_offset = 'cell_offset'
pbc = 'pbc'
