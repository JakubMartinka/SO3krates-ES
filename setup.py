from setuptools import setup, find_packages

setup(
    name="mlff",
    version="0.4.1",
    description="Build Neural Networks for Force Fields with JAX",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        # The JAX stack is pinned to the versions the test suite passes with: this is 2023-era code,
        # and newer jax/flax removed `jax.tree_map` and changed `jnp.array(None)`. For a GPU, install
        # the matching CUDA build of jax 0.4.28 on top (see README).
        "jax==0.4.28",
        "jaxlib==0.4.28",
        "flax==0.8.5",
        "optax==0.2.2",
        "chex<=0.1.90",
        # ml_dtypes 0.6 (pulled in by jax) requires numpy>=2, which would upgrade numpy underneath
        # MLatom (numpy<2); 0.5.x works with both.
        "ml_dtypes<0.6",
        "jaxopt",
        "jraph",
        "orbax-checkpoint == 0.5.23",
        "portpicker",
        # 'tensorflow',
        "scikit-learn",
        "ase",
        "tqdm",
        "wandb",
        "pyyaml",
        "h5py",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
    package_data={"": ["sph_ops/cgmatrix.npz",
                       "sph_ops/u_matrix.pickle"]},
    entry_points={
        "console_scripts": [
            "evaluate=mlff.cAPI.mlff_eval:evaluate",
            "train=mlff.cAPI.mlff_train:train",
            "run_md=mlff.cAPI.mlff_md:run_md",
            "run_relaxation=mlff.cAPI.mlff_structure_relaxation:run_relaxation",
            "analyse_md=mlff.cAPI.mlff_analyse:analyse_md",
            "train_so3krates=mlff.cAPI.mlff_train_so3krates:train_so3krates",
            "train_so3kratACE=mlff.cAPI.mlff_train_so3kratace:train_so3kratace",
            "trajectory_to_xyz=mlff.cAPI.mlff_postprocessing:trajectory_to_xyz",
            "to_mlff_input=mlff.cAPI.mlff_input_processing:to_mlff_input",
        ],
    },
)
