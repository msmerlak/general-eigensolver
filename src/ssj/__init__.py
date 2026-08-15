"""Simultaneous Saturated Jacobi (SSJ) eigensolver."""
from .core import ssj_eigh, off_frobenius
from .ipt import (ipt_eigh, ipt_eig, ipt_eig_partial, ipt_rate_columns,
                  ssj_ipt_eigh, refine_eig)
from .normal import normal_eig, normality_defect, shear_toward_normal
from .sdc import sdc_eigvals, matrix_sign
from .dispatch import eig_partial
from .block_ipt import (block_ipt_eig, adaptive_block_ipt_eig, choose_block,
                        sparse_block_ipt_eig)
from .purify import purify, spectral_projector, purify_split
from .gipt import gipt_eig
from .anderson import anderson_ipt_eig
from .davidson import davidson_eig
from .window import window_eig, window_count
from .riccati import bw_eig_partial

__all__ = ["ssj_eigh", "off_frobenius", "ipt_eigh", "ipt_eig", "ipt_eig_partial", "ipt_rate_columns",
           "ssj_ipt_eigh", "refine_eig",
           "normal_eig", "normality_defect", "shear_toward_normal",
           "sdc_eigvals", "matrix_sign", "eig_partial", "block_ipt_eig", "adaptive_block_ipt_eig",
           "choose_block", "sparse_block_ipt_eig", "purify", "spectral_projector",
           "purify_split", "gipt_eig", "anderson_ipt_eig",
           "davidson_eig", "window_eig", "window_count", "bw_eig_partial"]
__version__ = "0.1.0"
