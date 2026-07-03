'''kuant.options — option-specific analytics on top of kuant.core.

Contains:
  - Greeks (first-order): bscalldelta, bsputdelta
  - Greeks (second-order): bsgamma, bsvega
  - Greeks (rate sensitivity): bscallrho, bsputrho
  - Implied volatility solver: impvol
'''
from .bscallcharm import bscallcharm
from .bscalldelta import bscalldelta
from .bscallrho import bscallrho
from .bscalltheta import bscalltheta
from .bsgamma import bsgamma
from .bsputcharm import bsputcharm
from .bsputdelta import bsputdelta
from .bsputrho import bsputrho
from .bsputtheta import bsputtheta
from .bsvega import bsvega
from .impvol import impvol
from .impvolbisection import impvolbisection

__all__ = [
    'bscalldelta', 'bsputdelta',
    'bsgamma', 'bsvega',
    'bscallrho', 'bsputrho',
    'bscalltheta', 'bsputtheta',
    'bscallcharm', 'bsputcharm',
    'impvol', 'impvolbisection',
]
