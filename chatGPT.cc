#include <iostream>
#include <thread>
#include <vector>
#include <functional>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <future>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    std::shared_ptr<std::promise<void>> taskPromise;

                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) {
                            return;
                        }

                        task = std::move(this->tasks.front().first);
                        taskPromise = this->tasks.front().second;
                        this->tasks.pop();
                    }

                    task();  // 执行任务

                    // 如果任务是异步的，完成后通知
                    if (taskPromise) {
                        taskPromise->set_value();
                    }
                }
            });
        }
    }

    // 提交异步任务
    template <class F>
    std::shared_ptr<std::promise<void>> enqueue(F&& f) {
        auto taskPromise = std::make_shared<std::promise<void>>();
        auto task = [f, taskPromise] {
            f();
            taskPromise->set_value();  // 任务完成时设置值
        };

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::move(task), taskPromise);
        }
        condition.notify_one();  // 通知线程池有新任务
        return taskPromise;
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
    std::vector<std::thread> workers;
    std::queue<std::pair<std::function<void()>, std::shared_ptr<std::promise<void>>>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
};

void print_hello(int id) {
    std::this_thread::sleep_for(std::chrono::seconds(1));  // 模拟异步任务
    std::cout << "Hello from thread " << id << std::endl;
}

int main() {
    ThreadPool pool(3);  // 创建线程池，最大线程数为 3

    // 提交 5 个异步任务
    std::vector<std::shared_ptr<std::promise<void>>> promises;
    for (int i = 0; i < 5; ++i) {
        promises.push_back(pool.enqueue([i] { print_hello(i); }));
    }

    // 等待所有任务完成
    for (auto& promise : promises) {
        promise->get_future().get();  // 等待异步任务完成
    }

    return 0;
}
