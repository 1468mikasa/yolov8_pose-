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

int main(int argc, char **argv)
{
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
	// 没有连接设备
	if (iCameraCounts == 0)
	{
		return -1;
	}
	// 相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
	iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

	// 初始化失败
	printf("state = %d\n", iStatus);
	if (iStatus != CAMERA_STATUS_SUCCESS)
	{
		return -1;
	}

	// 获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
	CameraGetCapability(hCamera, &tCapability);

	//
	g_pRgbBuffer = (unsigned char *)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);
	// g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

	/*让SDK进入工作模式，开始接收来自相机发送的图像
	数据。如果当前相机是触发模式，则需要接收到
	触发帧以后才会更新图像。    */
	CameraPlay(hCamera);

	/*其他的相机参数设置
	例如 CameraSetExposureTime   CameraGetExposureTime  设置/读取曝光时间
		 CameraSetImageResolution  CameraGetImageResolution 设置/读取分辨率
		 CameraSetGamma、CameraSetConrast、CameraSetGain等设置图像伽马、对比度、RGB数字增益等等。
		 本例程只是为了演示如何将SDK中获取的图像，转成OpenCV的图像格式,以便调用OpenCV的图像处理函数进行后续开发
	*/

	if (tCapability.sIspCapacity.bMonoSensor)
	{
		channel = 1;
		CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
	}
	else
	{
		channel = 3;
		CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
	}

	const std::string model_path = "/home/auto/Desktop/yolov8_pose-/model/best_openvino_model/best.xml";
	// Define the confidence and NMS thresholds
	const float confidence_threshold = 0.4;
	const float NMS_threshold = 0.5;

	// Initialize the YOLO inference with the specified model and parameters
	yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

	// 循环显示1000帧图像
	double simage = 0;
	double time = 0;
	double result = 0;
	while (iDisplayFrames--)
	{


		if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		{
			CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);

			cv::Mat image(
				cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight),
				sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
				g_pRgbBuffer);

			// Check if the image was successfully loaded
			if (image.empty())
			{
				std::cerr << "ERROR: image is empty" << std::endl;
				return 1;
			}
	
	auto s = std::chrono::high_resolution_clock::now();

			inference.Pose_Run_async_Inference(image);

			

			//std::cerr << "@@@@@ 输入！@@@@" << std::endl;
	

			CameraReleaseImageBuffer(hCamera, pbyBuffer);

		auto e = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> diff = e - s;

		time += diff.count();  
		std::cout<<"time:"<<time<<std::endl;
			simage+=1;

		}


		std::cout<<"simage 画面:"<<simage<<std::endl;
	}

	CameraUnInit(hCamera);
	// 注意，现反初始化后再free
	free(g_pRgbBuffer);

	return 0;
}