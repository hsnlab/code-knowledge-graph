from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class CfgNodeType(Enum):
    """CFG Node Types"""
    FUNCTION_DECLARATION = "Function"
    ENTRY = "Entry"
    EXIT = "Exit"
    BASIC_BLOCK = "BasicBlock"
    CALL = "Call"
    BRANCH = "Branch"
    MERGE = "Merge"

@dataclass
class CfgNode:
    """Represents a node in the Control Flow Graph"""
    node_id: int
    code: str
    type: CfgNodeType
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'node_id': self.node_id,
            'code': self.code,
            'type': self.type.value
        }