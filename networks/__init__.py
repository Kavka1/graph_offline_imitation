# Register Network Classes here.
from .base import ActorCriticPolicy, ActorValuePolicy, ActorCriticValuePolicy, ActorPolicy
from .mlp import (
    ContinuousMLPActor,
    ContinuousMLPCritic,
    DiagonalGaussianMLPActor,
    MLPValue,
    MLPEncoder,
    DiscreteMLPDistance,
    MLPDiscriminator,
)
from .drqv2             import DrQv2Encoder, DrQv2Critic, DrQv2Actor, DrQv2Value, DiscreteDrQv2Distance, DrQv2Discriminator
from .rmimic            import RobomimicEncoder
from .gofar             import GoFarNetwork
from .contrastive       import ContrastiveGoalCritic, ContrastiveRLNetwork
from .contrastiveoi     import ContrastiveV, ContrastiveQ, ContrastiveOfflineImitationNetwork
from .contrastiveoi_v2  import ContrastiveOfflineImitationV2Network
from .dwbc              import DWBCDiscriminator, DWBCNetwork
from .smodice           import SMODICENetwork, SMODICE_Discriminator
from .oril              import ORILNetwork, ORILDiscriminator
from .bc                import BCNetwork