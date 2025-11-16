from package.adapters import CfgAdapter 
from package.adapters import LanguageAstAdapterRegistry


@LanguageAstAdapterRegistry.register_cfg('erlang')
class ErlangCfgAdapter(CfgAdapter):
    pass