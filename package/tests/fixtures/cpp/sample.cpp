#include <iostream>
#include <vector>
#include <string>
#include "myheader.h"
#include "utils/helper.h"
#include <algorithm>

class MyClass {       // class
public:
    void sayHello() { // function
        std::cout << "Hello from MyClass!" << std::endl;
    }
};

int main() {
    MyClass obj;
    obj.sayHello();   // call
    return 0;
}

main();