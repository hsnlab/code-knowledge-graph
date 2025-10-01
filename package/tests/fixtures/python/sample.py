import sys
import functools as fun

# class
class MyClass:
    # function
    def say_hello(self):
        print("Hello from MyClass!")


# function with call
def main():
    obj = MyClass()
    obj.say_hello()  # call


main()  # call