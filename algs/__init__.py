from .dwsl          import DWSL
from .gofar         import GoFar
from .sac           import SAC
from .goplan        import GoPlan
from .goplanv2      import GoPlanV2
from .goplanv3      import GoPlanV3
from .contrastive   import ContrastiveRL

from .offlineimitation.contrastiveoi    import ContrastiveOfflineImitation
from .offlineimitation.contrastiveoi_v2 import ContrastiveOfflineImitationV2
from .offlineimitation.dwbc             import DWBCOfflineImitation
from .offlineimitation.smodice          import SMODiceOfflineImitation

VALID_ALGORITHMS = [
    'sac', 'dwsl', 'gofar', 'goplan', 'goplanv2', 'goplanv3', 
    'contrastive', 
    'contrastiveoi', 'dwbc', 'contrastive_v2', 'smodice'
]