#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count = 4) : stop(false) {
        for (size_t i = 0; i < thread_count; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        using return_type = decltype(f(args...));
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};



int main(int argc, char** argv) {
    // ... 相机初始化代码 ...

    // 初始化线程池（4线程）
    ThreadPool thread_pool(4);

    // 定义互斥锁保护images队列
    std::mutex images_mutex;

    // ... 其他变量定义 ...
}

while (iDisplayFrames--) {
    auto s = std::chrono::high_resolution_clock::now();

    // 异步获取图像（保持原有逻辑）
    std::future<void> get_image_future = std::async(std::launch::async, [&]() {
        // ... 图像采集代码 ...
        });

    // 使用线程池处理推理任务
    {
        std::lock_guard<std::mutex> lock(images_mutex); // 加锁
        if (images.size() >= 2) {
            // 提交任务到线程池
            thread_pool.enqueue([&inference, &images] {
                inference.Pose_Run_async_Inference(images[0]);
                });
            thread_pool.enqueue([&Ainference, &images] {
                Ainference.Pose_Run_async_Inference(images[1]);
                });

            // 删除已处理的图像
            images.erase(images.begin(), images.begin() + 2);
            shanchu += 2;
        }
    }

    // ... 剩余代码（计时、打印等） ...
}


// 初始化4个推理实例
yolo::Inference inference1(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
yolo::Inference inference2(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
yolo::Inference inference3(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
yolo::Inference inference4(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

// 在循环中动态分配任务
{
    std::lock_guard<std::mutex> lock(images_mutex);
    for (size_t i = 0; i < images.size(); i++) {
        if (i % 4 == 0) {
            thread_pool.enqueue([&inference1, &images, i] {
                inference1.Pose_Run_async_Inference(images[i]);
                });
        }
        else if (i % 4 == 1) {
            thread_pool.enqueue([&inference2, &images, i] {
                inference2.Pose_Run_async_Inference(images[i]);
                });
        }
        else if (i % 4 == 2) {
            thread_pool.enqueue([&inference3, &images, i] {
                inference3.Pose_Run_async_Inference(images[i]);
                });
        }
        else if (i % 4 == 3) {
            thread_pool.enqueue([&inference4, &images, i] {
                inference4.Pose_Run_async_Inference(images[i]);
                });
        }
    }
    // 清空已处理图像
    images.clear();
}


set(CMAKE_CXX_STANDARD 11)