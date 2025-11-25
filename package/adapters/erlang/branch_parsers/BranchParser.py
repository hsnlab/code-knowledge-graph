from abc import ABC, abstractmethod
from tree_sitter import Tree, Node
from package.adapters.erlang.CfgNodeType import CfgNodeType, CfgNode

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from package.adapters.erlang.ErlangCfgParser import ErlangCfgParser

class BranchParser(ABC):

    def __init__(self, parser: 'ErlangCfgParser'):
        self.parser = parser

    @abstractmethod
    def parse(self, branch_node: Node, entry_node: CfgNode, code: str) -> CfgNode:
        pass 