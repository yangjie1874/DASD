import cv2
import os

def crop_center(image, crop_size=(256, 256)):
    """
    从图像中心裁剪指定大小的图像。

    :param image: 输入的图像
    :param crop_size: 裁剪的大小 (width, height)
    :return: 裁剪后的图像
    """
    h, w, _ = image.shape
    new_w, new_h = crop_size

    # 计算中心位置
    center_x, center_y = w // 2, h // 2

    # 计算裁剪区域的边界
    left = max(center_x - new_w // 2, 0)
    right = min(center_x + new_w // 2, w)
    top = max(center_y - new_h // 2, 0)
    bottom = min(center_y + new_h // 2, h)

    # 裁剪图像
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def process_images(input_folder, output_folder, crop_size=(416, 416)):
    """
    处理文件夹中的图像，将其中心裁剪为指定大小，并保存到输出文件夹。

    :param input_folder: 输入图像文件夹路径
    :param output_folder: 输出图像文件夹路径
    :param crop_size: 裁剪的大小 (width, height)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is not None:
                cropped_img = crop_center(img, crop_size)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_img)

if __name__ == "__main__":
    input_folder = 'datasets/wheat3/source'  # 输入图像文件夹路径
    output_folder = 'datasets/wheat3/source2'  # 输出图像文件夹路径

    process_images(input_folder, output_folder)