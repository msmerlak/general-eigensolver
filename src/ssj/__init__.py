"""Simultaneous Saturated Jacobi (SSJ) eigensolver."""
from .core import ssj_eigh, off_frobenius
from .ipt import ipt_eigh, ipt_eig, ssj_ipt_eigh
from .normal import normal_eig, normality_defect, shear_toward_normal

__all__ = ["ssj_eigh", "off_frobenius", "ipt_eigh", "ipt_eig", "ssj_ipt_eigh",
           "normal_eig", "normality_defect", "shear_toward_normal"]
__version__ = "0.1.0"
