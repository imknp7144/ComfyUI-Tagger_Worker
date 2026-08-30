from .node_camie  import CamieTaggerWorkerNode
from .node_anima  import AnimaTaggerWorkerNode
from .node_wd14   import WD14TaggerWorkerNode
from .node_joytag import JoyTagTaggerWorkerNode
from .node_mosaic import NSFWMosaicWorkerNode
from .node_isnet  import IsnetAnimeWorkerNode
from .node_setup  import SetupTaggerWorkerEnvironmentNode

# tagger_ensemble_worker由来6タガー(OpenVINO/iGPU経由)
from .node_cl_v1      import CLTaggerV1WorkerNode
from .node_cl_v2      import CLTaggerV2WorkerNode
from .node_dtq_l16    import DTQL16WorkerNode
from .node_dtq_b16    import DTQB16WorkerNode
from .node_oppai_v11  import OppaiOracleV11WorkerNode
from .node_wd_eva02_l import WDEva02LargeWorkerNode

NODE_CLASS_MAPPINGS = {
    "CamieTaggerWorker":              CamieTaggerWorkerNode,
    "AnimaTaggerWorker":              AnimaTaggerWorkerNode,
    "WD14TaggerWorker":               WD14TaggerWorkerNode,
    "JoyTagTaggerWorker":             JoyTagTaggerWorkerNode,
    "NSFWMosaicWorker":               NSFWMosaicWorkerNode,
    "IsnetAnimeWorker":               IsnetAnimeWorkerNode,
    "SetupTaggerWorkerEnvironment":   SetupTaggerWorkerEnvironmentNode,
    "CLTaggerV1Worker":               CLTaggerV1WorkerNode,
    "CLTaggerV2Worker":               CLTaggerV2WorkerNode,
    "DTQL16Worker":                   DTQL16WorkerNode,
    "DTQB16Worker":                   DTQB16WorkerNode,
    "OppaiOracleV11Worker":           OppaiOracleV11WorkerNode,
    "WDEva02LargeWorker":             WDEva02LargeWorkerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerWorker":              "Camie Tagger (Worker)",
    "AnimaTaggerWorker":              "Anima Tagger (Worker)",
    "WD14TaggerWorker":               "WD14 Tagger (Worker)",
    "JoyTagTaggerWorker":             "JoyTag Tagger (Worker)",
    "NSFWMosaicWorker":               "NSFW Mosaic (Worker)",
    "IsnetAnimeWorker":               "isnet-anime Segmentation (Worker)",
    "SetupTaggerWorkerEnvironment":   "Setup Tagger Worker Environment",
    "CLTaggerV1Worker":               "CL Tagger v1 (Ensemble/iGPU)",
    "CLTaggerV2Worker":               "CL Tagger v2 (Ensemble/iGPU)",
    "DTQL16Worker":                   "DanbooruTagQuery L16 (Ensemble/iGPU)",
    "DTQB16Worker":                   "DanbooruTagQuery B16 (Ensemble/iGPU)",
    "OppaiOracleV11Worker":           "OppaiOracle v1.1 (Ensemble/iGPU)",
    "WDEva02LargeWorker":             "WD EVA02-Large v3 (Ensemble/iGPU)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
