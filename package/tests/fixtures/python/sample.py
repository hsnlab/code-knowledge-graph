import sys
import os
import functools as fun
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
import math


# Simple class - no inheritance
class SimpleClass:
    def method_one(self):
        pass

    def method_with_params(self, name: str, age: int) -> str:
        """Method with type hints and docstring"""
        return f"{name} is {age}"

    def method_no_hints(self, x, y, z):
        """Method without type hints"""
        return x + y + z


# Single inheritance
class BaseClass:
    def base_method(self):
        pass

    @staticmethod
    def static_method(value: int) -> int:
        """Static method"""
        return value * 2

    @classmethod
    def class_method(cls, data: str):
        """Class method"""
        return cls()


class SingleInheritance(BaseClass):
    def child_method(self):
        pass

    def override_method(self, param: Optional[str] = None) -> bool:
        """Method with default parameter"""
        return param is not None


# Multiple inheritance
class Mixin:
    def mixin_method(self):
        pass


class MultipleInheritance(BaseClass, Mixin):
    def combined_method(self):
        pass

    def complex_params(self, *args, **kwargs) -> None:
        """Method with *args and **kwargs"""
        pass


# Original class
class MyClass:
    def say_hello(self):
        print('Hello from MyClass!')

    def nested_function_example(self):
        """Method with nested function"""

        def inner_helper(x: int) -> int:
            return x * 2

        return inner_helper(5)


# Standalone functions
def simple_function():
    """Simple function with no params"""
    pass


def typed_function(name: str, age: int, active: bool = True) -> Dict[str, any]:
    """Function with type hints and default value"""
    return {"name": name, "age": age, "active": active}


def no_type_hints(a, b, c):
    """Function without type hints"""
    return a + b + c


def complex_signature(pos_only, /, normal, *, kw_only, default=10) -> Tuple[int, int]:
    """Function with positional-only and keyword-only params"""
    return (pos_only, kw_only)


def main():
    obj = MyClass()
    obj.say_hello()

