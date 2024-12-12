from PIL import Image
import os

# 设置输入和输出路径
input_dir = 'datasets/wheat4/target'
output_dir = 'datasets/wheat4/target'

# 创建输出目录(如果不存在)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有JPG文件
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # 构建输入和输出文件路径
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')

        # 打开图像并保存为PNG格式
        image = Image.open(input_path)
        image.save(output_path, 'PNG')
        print(f'Converted {filename} to PNG')