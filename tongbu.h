#include <iostream>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        // 创建线程池
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        // 等待任务队列不为空
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();  // 执行任务
                }
            });
        }
    }

    // 提交任务
    template <class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();  // 通知线程池有新任务
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();  // 通知所有线程停止
        for (std::thread& worker : workers) {
            worker.join();  // 等待所有线程完成
        }
    }

private:
    std::vector<std::thread> workers;            // 存储工作线程
    std::queue<std::function<void()>> tasks;     // 存储任务队列
    std::mutex queueMutex;                       // 用于保护任务队列的互斥锁
    std::condition_variable condition;           // 用于线程同步的条件变量
    std::atomic<bool> stop;                      // 控制线程池是否停止
};


