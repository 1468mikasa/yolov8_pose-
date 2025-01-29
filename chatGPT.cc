int main(int argc, char** argv) {
    int iCameraCounts = 1;
    int iStatus = -1;
    tSdkCameraDevInfo tCameraEnumList;
    int hCamera;
    tSdkCameraCapbility tCapability; // 设备描述信息
    tSdkFrameHead sFrameInfo;
    BYTE* pbyBuffer;
    int iDisplayFrames = 10000;
    IplImage* iplImage = NULL;
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
    g_pRgbBuffer = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);

    // 开始接收图像数据
    CameraPlay(hCamera);

    // 初始化模型
    const std::string model_path = "/home/auto/Desktop/yolov8_pose-/model/best_openvino_model/best.xml";
    const float confidence_threshold = 0.2;
    const float NMS_threshold = 0.5;

    yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
    yolo::Inference Ainference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

    // 创建线程池
    ThreadPool pool(std::thread::hardware_concurrency()); // 创建一个线程池，使用CPU核心数量

    double simage = 0;
    double time = 0;
    double shanchu = 0;

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
                    std::lock_guard<std::mutex> lock(queueMutex);
                    taskQueue.push(image);
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

        // 将推理任务提交给线程池
        pool.enqueue([&inference, &Ainference] {
            // 从队列中获取图像进行推理
            std::unique_lock<std::mutex> lock(queueMutex);
            if (!taskQueue.empty()) {
                cv::Mat frame = taskQueue.front();
                taskQueue.pop();
                lock.unlock();

                // 进行推理
                inference.Pose_Run_async_Inference(frame);
                Ainference.Pose_Run_async_Inference(frame);
            }
            });

        std::cout << "相机帧:" << simage / time * 1000 << std::endl;
        std::cout << "推理帧:" << shanchu / time * 1000 << std::endl;
    }

    // 停止所有线程
    stopThreads = true;
    queueCV.notify_all();

    CameraUnInit(hCamera);
    free(g_pRgbBuffer);

    return 0;
}
