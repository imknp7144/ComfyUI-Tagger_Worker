from .node_camie  import CamieTaggerWorkerNode
from .node_anima  import AnimaTaggerWorkerNode
from .node_wd14   import WD14TaggerWorkerNode
from .node_joytag import JoyTagTaggerWorkerNode
from .node_mosaic import NSFWMosaicWorkerNode
from .node_setup  import SetupTaggerWorkerEnvironmentNode

NODE_CLASS_MAPPINGS = {
    "CamieTaggerWorker":              CamieTaggerWorkerNode,
    "AnimaTaggerWorker":              AnimaTaggerWorkerNode,
    "WD14TaggerWorker":               WD14TaggerWorkerNode,
    "JoyTagTaggerWorker":             JoyTagTaggerWorkerNode,
    "NSFWMosaicWorker":               NSFWMosaicWorkerNode,
    "SetupTaggerWorkerEnvironment":   SetupTaggerWorkerEnvironmentNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamieTaggerWorker":              "Camie Tagger (Worker)",
    "AnimaTaggerWorker":              "Anima Tagger (Worker)",
    "WD14TaggerWorker":               "WD14 Tagger (Worker)",
    "JoyTagTaggerWorker":             "JoyTag Tagger (Worker)",
    "NSFWMosaicWorker":               "NSFW Mosaic (Worker)",
    "SetupTaggerWorkerEnvironment":   "Setup Tagger Worker Environment",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
