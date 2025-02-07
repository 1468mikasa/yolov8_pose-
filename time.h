#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

class PeriodicPrinter {
public:
    PeriodicPrinter() : running(true) {
        // 启动打印线程
        printer_thread = std::thread(&PeriodicPrinter::print, this);
    }

    ~PeriodicPrinter() {
        // 停止打印线程
        running = false;
        if (printer_thread.joinable()) {
            printer_thread.join();
        }
    }

private:
    void print() {
        while (running) {
            std::cout << "———10ms———" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::thread printer_thread;
    std::atomic<bool> running;
};



