import unittest
import json
import os
from package.ast_processor import AstProcessor
from package.adapters import adapter_mapper, LanguageAdapter
from pandas.testing import assert_frame_equal
from pandas import DataFrame, notna

import re

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

    def _test_language_classes(self, language):
        """Generic test method for any language"""
        # Get config
        config = self.test_config[language]

        processor: AstProcessor = self.processors[language]

        expected_list = config['expected_classes']
        expected_df = DataFrame(expected_list)
        if len(expected_list) == 0:
            expected_df = DataFrame(columns=['file_id', 'cls_id', 'name', 'base_classes'])

        assert_frame_equal(processor.classes, expected_df,
                           obj=f"{language} classes")

    def _test_language_functions(self, language):
        """Generic test method for any language"""
        # Get config
        config = self.test_config[language]

        processor: AstProcessor = self.processors[language]

        expected_list = config['expected_functions']
        expected_df = DataFrame(expected_list)
        if len(expected_list) == 0:
            expected_df = DataFrame(columns=["file_id", "fnc_id", "name", "class", "class_base_classes", "params",
                                             "docstring", "function_code", "class_id", "return_type"
                                             ])

        def normalize_function_code_whitespace(df):
            """Normalize whitespace and quotes in function_code column"""
            df_copy = df.copy()
            if 'function_code' in df_copy.columns:
                df_copy['function_code'] = df_copy['function_code'].apply(
                    lambda x: re.sub(r'\s+', ' ', str(x).replace('\\n', '\n').replace('"', "'")).strip() if notna(
                        x) else x
                )
            return df_copy

        actual_normalized = normalize_function_code_whitespace(processor.functions)
        expected_normalized = normalize_function_code_whitespace(expected_df)

        assert_frame_equal(actual_normalized, expected_normalized,
                           check_dtype=False,
                           obj=f"{language} functions")

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


        test_method = lambda self, lang=language: self._test_language_classes(lang)
        test_method.__name__ = f'test_{language}_class'
        test_method.__doc__ = f'Test {language} class extraction'
        setattr(TestAstProcessor, test_method.__name__, test_method)

        test_method = lambda self, lang=language: self._test_language_functions(lang)
        test_method.__name__ = f'test_{language}_function'
        test_method.__doc__ = f'Test {language} function extraction'
        setattr(TestAstProcessor, test_method.__name__, test_method)

    # Store processors dict on the class
    TestAstProcessor.processors = processors


generate_tests()

if __name__ == '__main__':
    unittest.main(verbosity=2)