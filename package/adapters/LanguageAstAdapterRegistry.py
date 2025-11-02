from package.adapters import LanguageAstAdapter
from package.adapters import CfgAdapter


class LanguageAstAdapterRegistry:
    language_adapter_mapper: dict[str, LanguageAstAdapter] = {}
    language_cfg_mapper: dict[str, CfgAdapter] = {}
    
    @classmethod
    def register(cls, language: str):
        """Decorator to register an AST adapter for a specific language."""
        def decorator(adapter_class):
            cls.language_adapter_mapper[language.lower()] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def register_cfg(cls, language: str):
        """Decorator to register a CFG adapter for a specific language."""
        def decorator(adapter_class):
            cls.language_cfg_mapper[language.lower()] = adapter_class
            return adapter_class
        return decorator
    
    @classmethod
    def get_adapter(cls, language: str):
        """Get the AST adapter class for a specific language."""
        adapter = cls.language_adapter_mapper.get(language.lower())
        if adapter is None:
            raise ValueError(f"No AST adapter registered for language: {language}")
        return adapter
    
    @classmethod
    def get_cfg_adapter(cls, language: str):
        """Get the CFG adapter class for a specific language."""
        adapter = cls.language_cfg_mapper.get(language.lower())
        if adapter is None:
            raise ValueError(f"No CFG adapter registered for language: {language}")
        return adapter