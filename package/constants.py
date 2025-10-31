
# Mapper storing language → file extension pairs, used to determine which parser to apply.
# Must be expanded when adding support for a new language.

EXTENSION_MAP: dict[str, list[str]] = {
    "python": [".py", ".pyw", ".pyi"],
    "cpp": [".cpp", ".cc", ".cxx", ".c++", ".C",
            ".hpp", ".hh", ".hxx", ".h++", ".H", ".h"],
    "erlang": [".erl", ".hrl"],
}

# Reverse mapper for easier lookups

REVERSE_EXTENSION_MAP: dict[str, str] = {extension: lang for lang, extensions in EXTENSION_MAP.items() for extension in extensions}

# File extensions for project-defining configuration, documentation, and tooling files
# These catch most config files by their extension (e.g., .yml catches docker-compose.yml, rebar.config, etc.)
extensions = {
    '.md', '.txt', '.rst', '.ini', '.cfg', '.conf', '.toml', '.yaml', '.yml',
    '.json', '.xml', '.env', '.editorconfig',
    '.gitignore', '.gitattributes', '.gitmodules',
    '.dockerignore',
    '.clang-format', '.clang-tidy',
    '.pylintrc', '.flake8',
    '.app.src', '.app',
}

# Special filenames without extensions or with non-standard naming conventions
# These files need explicit matching as they can't be caught by extension alone
special_filenames = {
    'Dockerfile',
    'Jenkinsfile',
    'requirements.txt',
    'Pipfile',
    'MANIFEST.in',
    'Makefile', 'makefile', 'CMakeLists.txt', 'CMakeCache.txt',
    'conanfile.py',
    'Emakefile'
}