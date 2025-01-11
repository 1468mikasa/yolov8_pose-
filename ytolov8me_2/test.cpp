#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void Detector(std::string video_path, YOLO_V8*& p) {
    cv::VideoCapture cap("images\\1.mp4"); // 替换为你的视频文件路径

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    while(true)
    {


        cv::Mat img;
        cap >> img;
        if (img.empty()) {
            break; // 如果帧为空，则退出循环
        }

            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(155,255,155);

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );


            }
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(1);
        
        
    }
}


void Classifier(std::string video_path,YOLO_V8*& p)
{
    cv::VideoCapture cap(video_path); // 替换为你的视频文件路径

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    while (true)
    {
            cv::Mat img;
            cap >> img; // 读取下一帧
            if (img.empty()) {
                break; // 如果帧为空，则退出循环
            }
           
            std::vector<DL_RESULT> res;
            char* ret = p->RunSession(img, res);

            float positionY = 50;
            for (int i = 0; i < res.size(); i++)
            {
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                positionY += 50;
            }

            cv::imshow("TEST_CLS", img);
            cv::waitKey(1);
        

    }
}



int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}


void DetectTest()
{
#define USE_CUDA
    YOLO_V8* yoloDetector = new YOLO_V8;
    // ReadCocoYaml(yoloDetector);
    yoloDetector->classes = { "O","B","R" };
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.3;
    params.iouThreshold = 0.5;
    params.modelPath = "models/best2.onnx";
    params.imgSize = { 320, 320 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    yoloDetector->CreateSession(params);
    Detector("G:\\C++\\1\\ytolov8me_2\\ytolov8me_2\\Release\\images\\1_red.mp4", yoloDetector);
}


void ClsTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "cls.onnx";
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
    yoloDetector->CreateSession(params);
    Classifier("images/1_blue.mp4", yoloDetector);
}


int main()
{

    DetectTest();
 
}
