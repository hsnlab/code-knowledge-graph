#include <iostream>
#include <vector>
#include <string>
#include "myheader.h"
#include "utils/helper.h"
#include <algorithm>

// Simple class - no inheritance
class SimpleClass {
public:
    void simpleMethod() {
        std::cout << "Simple" << std::endl;
    }
};

// Base class
class BaseClass {
public:
    virtual void baseMethod() {
        std::cout << "Base" << std::endl;
    }
};

// Single inheritance
class SingleInheritance : public BaseClass {
public:
    void childMethod() {
        std::cout << "Child" << std::endl;
    }
};

// Multiple inheritance
class Mixin {
public:
    void mixinMethod() {
        std::cout << "Mixin" << std::endl;
    }
};

class MultipleInheritance : public BaseClass, private Mixin {
public:
    void combinedMethod() {
        std::cout << "Combined" << std::endl;
    }
};

// Template class
template<typename T>
class TemplateClass {
public:
    T data;
    void display() {
        std::cout << data << std::endl;
    }
};

// Struct with inheritance
struct SimpleStruct : public BaseClass {
    int value;
};

// Original class
class MyClass {
public:
    void sayHello() {
        std::cout << "Hello from MyClass!" << std::endl;
    }
};

int main() {
    MyClass obj;
    obj.sayHello();
    return 0;
}

main();