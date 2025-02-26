#include <iostream>
#include <thread>

void async_task(int task_id) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Task " << task_id << " done (Thread ID: " 
              << std::this_thread::get_id() << ")\n";
}

int main() {
    int task_id = 0;
    while (true) {
        // 每次循环启动一个线程并分离
        std::thread([task_id]() { async_task(task_id); }).detach();
        task_id++;
        
        // 主线程继续执行其他逻辑
        std::cout << "Main loop continues..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 控制任务生成速度
    }
    return 0;
}