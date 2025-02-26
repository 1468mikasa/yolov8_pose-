#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this]() { return !tasks.empty() || stop; });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F>
    void enqueue(F&& task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(task));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

int main() {
    ThreadPool pool(4);  // 4个工作线程
    int task_id = 0;

    while (true) {
        auto start_time = std::chrono::steady_clock::now();

        // 提交任务到线程池（非阻塞）
        pool.enqueue([task_id]() {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "Task " << task_id << " done\n";
        });
        task_id++;

        // 主线程逻辑
        std::cout << "Main loop continues..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 计算耗时
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        std::cout << "Loop iteration took: " << duration << " ms\n";
    }
    return 0;
}