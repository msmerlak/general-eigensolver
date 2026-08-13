"""Simultaneous Saturated Jacobi (SSJ) eigensolver."""
from .core import ssj_eigh, off_frobenius
from .ipt import (ipt_eigh, ipt_eig, ipt_eig_partial, ipt_rate_columns,
                  ssj_ipt_eigh, refine_eig)
from .normal import normal_eig, normality_defect, shear_toward_normal
from .sdc import sdc_eigvals, matrix_sign
from .dispatch import eig_partial

__all__ = ["ssj_eigh", "off_frobenius", "ipt_eigh", "ipt_eig", "ipt_eig_partial", "ipt_rate_columns",
           "ssj_ipt_eigh", "refine_eig",
           "normal_eig", "normality_defect", "shear_toward_normal",
           "sdc_eigvals", "matrix_sign", "eig_partial"]
__version__ = "0.1.0"
