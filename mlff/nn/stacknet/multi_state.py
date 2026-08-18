import jax
import jax.numpy as jnp

from typing import Any, Dict

from mlff.properties import property_names as pn

StackNet = Any


class MultiStateStackNet:
    """
    Evaluate a state-conditioned `StackNet` once per electronic state and stack the results, so a
    *multi-state* model presents exactly the same interface -- and the same output shapes -- as the
    *multi-output* one.

    The two variants differ only in how the `n_states` energies are produced:

      * **multi-output** (the original): one forward pass, `nn.Energy(n_states=N)` emitting N values
        from N independent linear heads on a shared backbone.
      * **multi-state** (this, the So3krates form of MS-ANI): one weight-shared network with the
        state index as an input (`nn.StateEmbed`) and a single-output energy head, evaluated N times.
        Every parameter is shared between states, and the state count is not baked into any weight
        shape -- so `n_states` here is a property of the *call*, not of the checkpoint.

    Wrapping rather than subclassing `nn.Module` keeps the checkpoint a plain `StackNet`
    (`init_stack_net` reconstructs it unchanged; the presence of `state_embed` among the feature
    embeddings is what marks it as multi-state), and keeps the parameter tree free of an extra
    vmap axis. All the observable-function factories in this package only ever touch `.apply` and
    `.prop_keys`, which is why a plain object suffices.

    The `n_states` passes are `jax.vmap`ed, so the energies come back with shape (n_states,) --
    identical to the multi-output variant -- and one `jax.jacrev` therefore still yields per-state
    forces of shape (n_states, n_atoms, 3). Everything downstream (the gap loss term, the energy
    shift bookkeeping, `predict`) is shape-compatible without changes. Cost is N times a
    multi-output pass, which is inherent to the architecture and is what MS-ANI pays as well.
    """

    def __init__(self, stack_net: StackNet, n_states: int):
        if n_states < 1:
            raise ValueError(f'n_states must be >= 1, got {n_states}')
        self.stack_net = stack_net
        self.n_states = n_states
        self.state_key = stack_net.prop_keys[pn.state]
        self.energy_key = stack_net.prop_keys[pn.energy]

    @property
    def prop_keys(self) -> Dict:
        return self.stack_net.prop_keys

    def _with_state(self, inputs: Dict, s) -> Dict:
        x = dict(inputs)
        x[self.state_key] = jnp.asarray(s, dtype=jnp.float32).reshape(1)
        return x

    def init(self, rng, inputs: Dict, *args, **kwargs):
        """Initialize the wrapped `StackNet`; its parameters are the whole parameter set."""
        return self.stack_net.init(rng, self._with_state(inputs, 0.), *args, **kwargs)

    def apply(self, params, inputs: Dict, *args, **kwargs) -> Dict[str, jnp.ndarray]:
        def single_state(s):
            return self.stack_net.apply(params, self._with_state(inputs, s), *args, **kwargs)

        states = jnp.arange(self.n_states, dtype=jnp.float32)  # shape: (n_states)
        out = jax.vmap(single_state)(states)  # every entry gains a leading state axis

        # The energy head has one output per pass, so stacking gives (n_states, 1); flatten it to
        # (n_states,), which is the shape nn.Energy(n_states=N) returns in the multi-output variant.
        out = dict(out)
        out[self.energy_key] = out[self.energy_key].reshape(-1)
        return out
