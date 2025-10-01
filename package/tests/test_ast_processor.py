import unittest
import json
import os
from package.ast_processor import AstProcessor
from package.adapters import adapter_mapper, LanguageAdapter
from pandas.testing import assert_frame_equal
from pandas import DataFrame

class TestAstProcessor(unittest.TestCase):
    """Test AstProcessor methods for different languages using tree-sitter"""

    @classmethod
    def setUpClass(cls):
        """Setup Phase - Load test configuration"""
        config_path = os.path.join(os.path.dirname(__file__), 'ast_test_config.json')
        with open(config_path, 'r') as f:
            cls.test_config = json.load(f)

    def setUp(self):
        """Setup Phase - Initialize parser"""
        self.parser = None  # Initialize your tree-sitter parser

    def _test_language_imports(self, language):
        """Generic test method for any language"""
        # Get config
        config = self.test_config[language]

        processor: AstProcessor = self.processors[language]

        expected_list = config['expected_imports']
        expected_df = DataFrame(expected_list)
        # Exercise: Parse and extract


        assert_frame_equal(processor.imports, expected_df,
                           obj=f"{language} imports")

    def tearDown(self):
        """Teardown Phase"""
        self.parser = None


# Generate test methods dynamically
def generate_tests():
    config_path = os.path.join(os.path.dirname(__file__), 'ast_test_config.json')
    with open(config_path, 'r') as f:
        test_config = json.load(f)

    # Store all processors in a class-level dict
    processors = {}

    for language in test_config.keys():
        config = test_config[language]
        adapter = adapter_mapper.get(language)
        file_path = os.path.join(os.path.dirname(__file__), config['path'])
        with open(file_path, 'rb') as f:
            file_content = f.read()

        processor = AstProcessor(adapter, file_content)
        processor.process_file_ast(None, {}, False)
        processors[language] = processor

        test_method = lambda self, lang=language: self._test_language_imports(lang)
        test_method.__name__ = f'test_{language}_imports'
        test_method.__doc__ = f'Test {language} import extraction'
        setattr(TestAstProcessor, test_method.__name__, test_method)

    # Store processors dict on the class
    TestAstProcessor.processors = processors


generate_tests()

if __name__ == '__main__':
    unittest.main(verbosity=2)