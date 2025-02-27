#include <iostream>
#include <thread>
#include <chrono>

// 需要在后台运行的循环任务
void background_task() {
    while (true) {  // 您的循环条件
        std::cout << "Background thread is running..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    // 创建并启动独立线程
    std::thread worker(background_task);
    
    // 将线程与主线程分离（使其在后台运行）
    worker.detach();

    // 主线程继续执行其他任务
    while (true) {
        std::cout << "Main thread is running..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}