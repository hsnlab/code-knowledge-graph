from package.adapters import LanguageAstAdapter

class LanguageAstAdapterRegistry:
    language_adapter_mapper: dict[str, LanguageAstAdapter] = {}

    @classmethod
    def register(cls, language: str):
        """Decorator to register an adapter for a specific language."""
        def decorator(adapter_class):
            cls.language_adapter_mapper[language.lower()] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def get_adapter(cls, language: str):
        """Get the adapter class for a specific language."""
        adapter = cls.language_adapter_mapper.get(language.lower())
        if adapter is None:
            raise ValueError(f"No adapter registered for language: {language}")
        return adapter
