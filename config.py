from enum import Enum

class GraphType(Enum):
    FULLY_CONNECTED = "fully_connected"
    THRESHOLD_BASED = "threshold_based"
    TOP_K = "top_k"