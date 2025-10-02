import sys
import os
import functools as fun
from collections import defaultdict, Counter
from typing import List, Dict
import math

# Simple class - no inheritance
class SimpleClass:
    def method_one(self):
        pass

# Single inheritance
class BaseClass:
    def base_method(self):
        pass

class SingleInheritance(BaseClass):
    def child_method(self):
        pass

# Multiple inheritance
class Mixin:
    def mixin_method(self):
        pass

class MultipleInheritance(BaseClass, Mixin):
    def combined_method(self):
        pass

# Original class
class MyClass:
    def say_hello(self):
        print("Hello from MyClass!")


def main():
    obj = MyClass()
    obj.say_hello()


main()