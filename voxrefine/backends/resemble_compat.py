"""Small compatibility shims for the pinned Resemble inference runtime."""

from __future__ import annotations


def install_numpy_fsolve_compat() -> None:
    """Adapt Resemble's scalar root conversion to NumPy 2.x array semantics."""
    import numpy as np
    import scipy
    from resemble_enhance.enhancer.lcfm.cfm import Solver

    def exponential_decay_mapping(t, n=4):
        def h(value, a):
            return (a**value - 1) / (a - 1)

        root = scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0)
        a = float(np.asarray(root).item())
        return h(t, a=a)

    Solver.exponential_decay_mapping = staticmethod(exponential_decay_mapping)
