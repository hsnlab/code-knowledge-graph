import unittest
import json
import os
from package.ast_processor import AstProcessor
from package.adapters import LanguageAstAdapterRegistry
from pandas.testing import assert_frame_equal
from pandas import DataFrame, notna
import sys
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

    def normalize_function_code_whitespace(self, df):
        """Normalize whitespace and quotes in function_code column"""
        df_copy = df.copy()
        if 'function_code' in df_copy.columns:
            df_copy['function_code'] = df_copy['function_code'].apply(
                lambda x: re.sub(r'\s+', ' ', str(x).replace('\\n', '\n').replace('"', "'")).strip() if notna(
                    x) else x
            )
        return df_copy

    def _test_specific_function_scenario(self, language_adapter_class, mock, expected):
        """

        :param language_adapter_class: Language specifict adapter, used to initiate object for parsing
        :param mock: Test script snippet
        :param expected: List containing the expected values for the given testcase
        :return:
        """
        processor = AstProcessor(language_adapter_class(), mock.encode('utf-8'))
        processor.process_file_ast(None, {}, False)
        expected_df = DataFrame(expected)
        if len(expected) == 0:
            expected_df = DataFrame(columns=["file_id", "fnc_id", "name", "class", "class_base_classes", "params",
                                             "docstring", "function_code", "class_id", "return_type"
                                             ])

        actual_normalized = self.normalize_function_code_whitespace(processor.functions)
        expected_normalized = self.normalize_function_code_whitespace(expected_df)

        assert_frame_equal(actual_normalized, expected_normalized,
                           check_dtype=False)

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

        actual_normalized = self.normalize_function_code_whitespace(processor.functions)
        expected_normalized = self.normalize_function_code_whitespace(expected_df)

        actual_str = processor.functions[['name', 'class']].to_string()
        assert_frame_equal(actual_normalized, expected_normalized,
                           check_dtype=False,
                           obj=f"Actual:\n{actual_str}")

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
        adapter_class = LanguageAstAdapterRegistry.get_adapter(language)
        adapter = adapter_class()
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

        function_tests = config.get("standalone_function_tests")
        if function_tests is not None:
            for test_name, test_data in function_tests.items():
                # Capture test_data in closure properly
                test_method = lambda self, adapter=adapter_class, mock=test_data.get("mock"),expected=test_data.get("expected"): \
                    self._test_specific_function_scenario(adapter, mock, expected)

                test_method.__name__ = f'test_{language}_{test_name}'
                test_method.__doc__ = f'Test {language} {test_name} function extraction'
                setattr(TestAstProcessor, test_method.__name__, test_method)


    # Store processors dict on the class
    TestAstProcessor.processors = processors


generate_tests()

if __name__ == '__main__':
    unittest.main(verbosity=2)