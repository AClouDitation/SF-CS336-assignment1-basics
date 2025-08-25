import cppyy

cppyy.cppdef(r'''
#include <future>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std;

int run_cpp_thread() {
    promise<int> p;
    future<int> f = p.get_future();

    thread t([&p]() {
        this_thread::sleep_for(10ms); // Simulate work
        p.set_value(2);
    });

    auto status = f.wait_for(10ms); // Wait for a short duration

    if (status == future_status::ready) {
        cout << "Task is ready" << endl;
    } else {
        cout << "Task is running or not yet ready" << endl;
    }

    t.join(); // Wait for the thread to complete
    return 0;
}
''')

# Call the C++ function from Python
cppyy.gbl.run_cpp_thread()