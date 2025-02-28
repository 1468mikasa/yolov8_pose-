#include "inference.h"
#include "CameraApi.h" // 相机SDK的API头文件
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <chrono>
using namespace cv;

unsigned char *g_pRgbBuffer; // 处理后数据缓存区
#include <iostream>
#include <opencv2/highgui.hpp>
#include <future>
#include <omp.h>

int main(int argc, char **argv)
{
	const std::string model_path = "/home/wei/桌面/yolov8_pose-/model/yolov8-f_best/best.xml";
	// Define the confidence and NMS thresholds
	const float confidence_threshold = 0.4;
	const float NMS_threshold = 0.5;

	// Initialize the YOLO inference with the specified model and parameters
	yolo::Inference inference(model_path, cv::Size(480, 480), confidence_threshold, NMS_threshold);

/* 	const std::string Amodel_path = "/home/auto/Desktop/yolov8_pose-/model/yolov8-f_best/best.xml";
	yolo::Inference Ainference(Amodel_path, cv::Size(640, 640), confidence_threshold, NMS_threshold); */
	// 循环显示1000帧图像
	double simage = 0;
	double time = 0;
	double result = 0;
	cv::Mat images;
	int flage = 0;
	images=cv::imread("/home/auto/Desktop/yolov8_pose-/22openvino/2.jpg");

	while (1)
	{
		
		auto start = std::chrono::high_resolution_clock::now();
		cv::waitKey(5);
		auto frame_ptr = std::make_shared<cv::Mat>(images);

		std::thread([frame_ptr, &inference]() {
			inference.Pose_RunInference(*frame_ptr);
		}).detach();
		


	
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> diff = end - start;
		simage += 1;
		time += diff.count();
		if (time > 1000)
		{
			auto result = (simage / time) * 1000;
			std::cout << "\n"
					  << std::endl;
			std::cout << result << "帧" << std::endl;
			std::cout << ((inference.huamianshu /* + Ainference.huamianshu */) / time) * 1000 << "处理帧" << std::endl;

			time = 0;
			simage = 0;
			inference.huamianshu = 0;
			//Ainference.huamianshu = 0;
		}
		
	}


	return 0;
}