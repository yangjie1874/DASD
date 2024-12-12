def convert_bbox_to_yolo(bbox, img_width=416, img_height=416):
    x_min, y_min, x_max, y_max, class_id = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return class_id, x_center, y_center, width, height


def process_line(line, img_width=416, img_height=416):
    parts = line.strip().split()
    img_path = parts[0]
    bboxes = [list(map(int, bbox.split(','))) for bbox in parts[1:]]
    yolo_bboxes = [convert_bbox_to_yolo(bbox, img_width, img_height) for bbox in bboxes]
    return img_path, yolo_bboxes


def write_yolo_format(file_path, yolo_bboxes):
    with open(file_path, 'w') as f:
        for bbox in yolo_bboxes:
            class_id, x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def main(input_file, output_dir):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip():
            img_path, yolo_bboxes = process_line(line)
            img_name = img_path.split('/')[-1].replace('.png', '.txt')
            output_file = f"{output_dir}/{img_name}"
            write_yolo_format(output_file, yolo_bboxes)


# Example usage
input_file = 'datasets/random_layout1/bounding_boxes.txt'
output_dir = '/media/bao511/18949309166/target1/label/'  # Replace with your desired output directory
main(input_file, output_dir)