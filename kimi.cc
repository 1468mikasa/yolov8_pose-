#include "inference.h"
#include "CameraApi.h" // 相机SDK的API头文件
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <chrono>
#include <future>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

using namespace cv;

unsigned char *g_pRgbBuffer; // 处理后数据缓存区

// 线程安全的任务队列
std::queue<cv::Mat> taskQueue;
std::mutex queueMutex;
std::condition_variable queueCV;
std::atomic<bool> stopThreads(false);
std::atomic<int> inference_count(0); // 推理帧数统计

// 推理线程函数
void inferenceThread(yolo::Inference& inference) {
    std::cout << "Inference thread started.void" << std::endl; 
    while (!stopThreads) {
        std::cout << "Inference thread started.while" << std::endl; 
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCV.wait(lock, [] { return !taskQueue.empty() || stopThreads; });

        if (stopThreads) {
            return;
        }

        // 从任务队列中取出一帧图像
        cv::Mat frame = taskQueue.front();
        taskQueue.pop();
        lock.unlock();

        try {
            // 运行异步推理
            std::cerr << "try" <<  std::endl;
            inference.Pose_Run_async_Inference(frame);
            inference_count++; // 更新推理帧数统计
        } catch (const std::exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char **argv) {
    int iCameraCounts = 1;
    int iStatus = -1;
    tSdkCameraDevInfo tCameraEnumList;
    int hCamera;
    tSdkCameraCapbility tCapability; // 设备描述信息
    tSdkFrameHead sFrameInfo;
    BYTE *pbyBuffer;
    int iDisplayFrames = 10000;
    IplImage *iplImage = NULL;
    int channel = 3;

    CameraSdkInit(1);

    // 枚举设备，并建立设备列表
    iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
    printf("state = %d\n", iStatus);

    printf("count = %d\n", iCameraCounts);
    if (iCameraCounts == 0) {
        return -1;
    }

    // 相机初始化
    iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);
    printf("state = %d\n", iStatus);
    if (iStatus != CAMERA_STATUS_SUCCESS) {
        return -1;
    }

    // 获得相机的特性描述结构体
    CameraGetCapability(hCamera, &tCapability);
    g_pRgbBuffer = (unsigned char *)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);

    // 开始接收图像数据
    CameraPlay(hCamera);
	CameraSetAeState(hCamera, false);
	CameraSetExposureTime(hCamera, 2500); 
    	CameraSetGain(hCamera, 255,255,255); 
    // 初始化模型
    const std::string model_path = "/home/auto/Desktop/yolov8_pose-/model/best_openvino_model/best.xml";
    const float confidence_threshold = 0.2;
    const float NMS_threshold = 0.5;

    yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
    yolo::Inference Ainference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    // 创建多个推理线程
    std::vector<std::thread> threads;
    threads.emplace_back(inferenceThread, std::ref(inference));
    threads.emplace_back(inferenceThread, std::ref(Ainference));
std::cout << "Inference threads created: " << threads.size() << std::endl;
    double simage = 0;
    double time = 0;

    while (iDisplayFrames--) {
        auto s = std::chrono::high_resolution_clock::now();

        std::future<void> get_image_future = std::async(std::launch::async, [&]() {
            if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);
                cv::Mat image(
                    cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight),
                    sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
                    g_pRgbBuffer);
                if (!image.empty()) {
                    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    std::lock_guard<std::mutex> lock(queueMutex);
                    taskQueue.push(image);
			        cv::imshow("AshowA",image);
                    cv::waitKey(1);
                    
                }
                simage += 1;
                CameraReleaseImageBuffer(hCamera, pbyBuffer);
            }
        });

        // 等待图像获取完成
        get_image_future.wait();

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = e - s;
        time += diff.count();
        std::cout << "diff.count():" << diff.count() << std::endl;

        std::cout << "相机帧:" << simage / time * 1000 << std::endl;
        std::cout << "推理帧:" << inference_count / time * 1000 << std::endl;
    }

    // 停止所有线程
    stopThreads = true;
    queueCV.notify_all();

    // 等待所有线程结束
    for (auto& thread : threads) {
        thread.join();
    }

    CameraUnInit(hCamera);
    free(g_pRgbBuffer);

    return 0;
}