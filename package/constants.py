
# Mapper storing language → file extension pairs, used to determine which parser to apply.
# Must be expanded when adding support for a new language.

EXTENSION_MAP: dict[str, list[str]] = {
    "python": [".py", ".pyw", ".pyi"],
    "cpp": [".cpp", ".cc", ".cxx", ".c++", ".C",
            ".hpp", ".hh", ".hxx", ".h++", ".H"],
    "erlang": [".erl", ".hrl"],
}

# Reverse mapper for easier lookups

REVERSE_EXTENSION_MAP: dict[str, str] = {extension: lang for lang, extensions in EXTENSION_MAP.items() for extension in extensions}

