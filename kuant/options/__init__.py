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
from .bscolor import bscolor
from .bsgamma import bsgamma
from .bsputcharm import bsputcharm
from .bsputdelta import bsputdelta
from .bsputrho import bsputrho
from .bsputtheta import bsputtheta
from .bsspeed import bsspeed
from .bsvanna import bsvanna
from .bsvega import bsvega
from .bsvolga import bsvolga
from .bszomma import bszomma
from .callpayoff import callpayoff
from .deltabucket import deltabucket
from .impvol import impvol
from .impvolbisection import impvolbisection
from .moneynessbucket import moneynessbucket
from .putpayoff import putpayoff

__all__ = [
    # first-order Greeks
    'bscalldelta', 'bsputdelta',
    'bsgamma', 'bsvega',
    'bscallrho', 'bsputrho',
    'bscalltheta', 'bsputtheta',
    'bscallcharm', 'bsputcharm',
    # second-order Greeks (put-call symmetric)
    'bsvanna', 'bsvolga',
    'bsspeed', 'bszomma', 'bscolor',
    # expiry payoffs
    'callpayoff', 'putpayoff',
    # chain-selection filters
    'deltabucket', 'moneynessbucket',
    # implied vol solvers
    'impvol', 'impvolbisection',
]
