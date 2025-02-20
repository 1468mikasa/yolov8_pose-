import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_images(label_folder, image_folder):
    # 物体类别
    class_list = ["person"]
    # 类别的颜色
    class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    # 关键点的顺序
    keypoint_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    # 关键点的颜色
    keypoint_color = [(255, 20, 100), (100, 255, 20), (100, 200, 255), (255, 255, 20), (255, 165, 20), (255, 255, 255), (20, 255, 255), (128, 122, 0), (220, 128, 0)]

    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".JPG"):
            img_path = os.path.join(image_folder, file_name)
            plt.figure(figsize=(15, 10))
            img = cv2.imread(img_path)
            plt.imshow(img[:, :, ::-1])
            plt.axis('off')

            # 构建对应的标签文件路径
            base_name = os.path.splitext(file_name)[0]
            yolo_txt_path = os.path.join(label_folder, base_name + ".txt")
            #print(yolo_txt_path)

            # 检查标签文件是否存在
            if os.path.exists(yolo_txt_path):
                with open(yolo_txt_path, 'r') as f:
                    lines = f.readlines()
                lines = [x.strip() for x in lines]
                label = np.array([x.split() for x in lines], dtype=np.float32)
                for l in label:
                    label_id = int(l[0])
                    if label_id >= len(class_list):
                        # 获取要移动的 txt 文件的路径
                        txt_file_path = os.path.join(label_folder, base_name + ".txt")
                        # 获取上级目录路径
                        parent_directory = os.path.dirname(label_folder)
                        # 构建目标路径
                        destination_path = os.path.join(parent_directory, base_name + ".txt")
                        try:
                            # 移动文件
                            os.rename(txt_file_path, destination_path)
                            print(f"已将 {txt_file_path} 移动到 {destination_path}")
                        except Exception as e:
                            print(f"移动文件时出错: {e}")
                img_copy = img.copy()
                h, w = img_copy.shape[:2]

                # 绘制检测框和关键点
                for id, l in enumerate(label):
                    # label_id,center x,y and width, height
                    label_id, cx, cy, bw, bh = l[0:5]
                    # 确保 label_id 在有效范围内
                    if 0 <= int(label_id) < len(class_list):
                        label_text = class_list[int(label_id)]
                    else:
                        print(f"Invalid label_id {label_id} in {yolo_txt_path}")
                        continue
                    # rescale to image size
                    cx *= w
                    cy *= h
                    bw *= w
                    bh *= h
                    # draw the bounding box
                    xmin = int(cx - bw / 2)
                    ymin = int(cy - bh / 2)
                    xmax = int(cx + bw / 2)
                    ymax = int(cy + bh / 2)
                    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), class_color[int(label_id)], 2)
                    cv2.putText(img_copy, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color[int(label_id)], 2)

                    # draw 17 keypoints, px,py,pv,px,py,pv...
                    for i in range(5, len(l), 3):
                        px, py, pv = l[i:i + 3]
                        # rescale to image size
                        px *= w
                        py *= h
                        # puttext the index
                        index = int((i - 5) / 3)
                        # draw the keypoints
                        cv2.circle(img_copy, (int(px), int(py)), 5, keypoint_color[index % len(keypoint_color)], -1)
                        keypoint_text = "{}_{}".format(index, keypoint_list[index])
                        # cv2.putText(img_copy, keypoint_text, (int(px), int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             keypoint_color[index % len(keypoint_color)], 1)

                plt.figure(figsize=(15, 10))
                plt.imshow(img_copy[:, :, ::-1])
                plt.axis('off')

                # 构建输出路径并保存图像
                out_path = os.path.join(image_folder, "processed_" + file_name)
                cv2.imwrite(f'{out_path}', img_copy)
                cv2.imshow("img", img_copy)
                cv2.waitKey(1000)
                plt.close()


# 示例用法
label_folder = r"F:\train\txt"
image_folder = r"F:\train\img"
process_images(label_folder, image_folder)