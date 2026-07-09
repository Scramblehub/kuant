"""kuant.options: option-specific analytics on top of kuant.core.

Contains:
  - First-order Greeks: bscalldelta, bsputdelta
  - Second-order Greeks: bsgamma, bsvega, bsvanna, bsvolga, bsspeed,
    bszomma, bscolor, bscallcharm, bsputcharm
  - Rate / carry sensitivities: bscallrho, bsputrho, bscalltheta, bsputtheta
  - Expiry payoffs and chain filters: callpayoff, putpayoff, deltabucket,
    moneynessbucket
  - Implied volatility solvers: impvol (Newton), impvolbisection
  - Exotic pricers (v0.6 batch 8): digitalprice, gapprice, lookbackprice,
    chooserprice, powerprice
"""

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
from .chooserprice import chooserprice
from .deltabucket import deltabucket
from .digitalprice import digitalprice
from .gapprice import gapprice
from .impvol import impvol
from .impvolbisection import impvolbisection
from .lookbackprice import lookbackprice
from .moneynessbucket import moneynessbucket
from .powerprice import powerprice
from .putpayoff import putpayoff

__all__ = [
    # first-order Greeks
    "bscalldelta",
    "bsputdelta",
    "bsgamma",
    "bsvega",
    "bscallrho",
    "bsputrho",
    "bscalltheta",
    "bsputtheta",
    "bscallcharm",
    "bsputcharm",
    # second-order Greeks (put-call symmetric)
    "bsvanna",
    "bsvolga",
    "bsspeed",
    "bszomma",
    "bscolor",
    # expiry payoffs
    "callpayoff",
    "putpayoff",
    # chain-selection filters
    "deltabucket",
    "moneynessbucket",
    # implied vol solvers
    "impvol",
    "impvolbisection",
    # v0.6.0 batch 8: exotic option pricing
    "digitalprice",
    "gapprice",
    "lookbackprice",
    "chooserprice",
    "powerprice",
]
