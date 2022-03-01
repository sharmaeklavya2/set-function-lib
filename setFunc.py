from collections import Mapping
from typing import Dict, Optional, Tuple
from typing_extensions import TypeAlias
# from typing import Any, Union
# from fractions import Fraction
import numpy as np
# from nptyping import NDArray

# ValueT = Union[int, float, Fraction]
ValueT = int
# ValueNDArray: TypeAlias = NDArray[Any, ValueT]
BoolMaskPair: TypeAlias = Tuple[bool, Optional[int], Optional[int]]
ValueNDArray: TypeAlias = np.ndarray


def log2floor(n: int) -> int:
    return len(bin(n)[3:])


def checkStrongMonotonicity(v: ValueNDArray) -> BoolMaskPair:
    n = log2floor(v.shape[0])
    # print('n:', n)
    assert v.shape == (1 << n,)
    xmask = np.arange(0, v.shape[0])
    for ymask in range(v.shape[0] - 1):
        zmask = xmask[(xmask & ymask) == ymask]  # zmask is superset of ymask
        zmask = zmask[zmask != ymask]  # zmask is a strict superset of ymask
        # print('ymask:', ymask, ', zmask:', zmask)
        minzmask = zmask[v[zmask].argmin()]
        vy, vzmin = v[ymask], v[minzmask]
        if vzmin <= vy:
            return (False, ymask, minzmask)
    return (True, None, None)


def checkSubsetProps(v: ValueNDArray) -> Mapping[str, BoolMaskPair]:
    n = log2floor(v.shape[0])
    assert v.shape == (1 << n,)
    xmask = np.arange(0, v.shape[0])
    properties = ['subadd', 'subm', 'supadd', 'supm']
    d: Dict[str, BoolMaskPair] = {p: (True, None, None) for p in properties}
    for ymask in range(v.shape[0]):
        capmask, cupmask = (xmask & ymask), (xmask | ymask)
        vxy = v + v[ymask]
        vxy_minus_vcupcap = vxy - v[cupmask] - v[capmask]
        if d['subm'][0] and not np.all(vxy_minus_vcupcap >= 0):
            d['subm'] = (False, ymask, vxy_minus_vcupcap.argmin())
        if d['supm'][0] and not np.all(vxy_minus_vcupcap <= 0):
            d['supm'] = (False, ymask, vxy_minus_vcupcap.argmax())
        disjoint = (capmask == 0)
        vxyd_minus_vcupd = (vxy - v[cupmask])[disjoint]
        xd = xmask[disjoint]
        if d['subadd'][0] and not np.all(vxyd_minus_vcupd >= 0):
            d['subadd'] = (False, ymask, xd[vxyd_minus_vcupd.argmin()])
        if d['supadd'][0] and not np.all(vxyd_minus_vcupd <= 0):
            d['supadd'] = (False, ymask, xd[vxyd_minus_vcupd.argmax()])
        if sum([d[p][0] for p in properties]) == 0:
            break
    return d
