from abc import ABC, abstractmethod
from tree_sitter import Tree, Node
from cfg_services.erlang.CfgNodeType import CfgNodeType, CfgNode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cfg_services.erlang.ErlangCfgParser import ErlangCfgParser

class BranchParser(ABC):

    def __init__(self, parser: 'ErlangCfgParser'):
        self.parser = parser

    @abstractmethod
    def parse(self, branch_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        pass 