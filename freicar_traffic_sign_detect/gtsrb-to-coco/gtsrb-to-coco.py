"""
Takes the downloaded and unzipped GTSRB dataset (download_dataset.sh script) and prepares its file and directory structure for the use with YOLOv5 (coco dataset format)
"""

import os
import shutil

from PIL import Image
import pandas as pd
import numpy as np

IN_DIR = 'GTSRB'
IN_TRAIN_DIR = IN_DIR + os.sep + 'Final_Training' + os.sep + 'Images'
IN_TEST_DIR = IN_DIR + os.sep + 'Final_Test' + os.sep + 'Images'
IN_TEST_GT_DIR = IN_DIR + os.sep + 'GTSRB_Final_Test_GT'

OUT_DIR = 'processed_dataset'
OUT_IMAGE_DIR = OUT_DIR + os.sep + 'images'
OUT_LABEL_DIR = OUT_DIR + os.sep + 'labels'
OUT_TRAIN_DIR_NAME = 'train_v1'
OUT_VAL_DIR_NAME = 'val_v1'
OUT_TEST_DIR_NAME = 'test_v1'

num_processed_train_img = 0
num_processed_test_img = 0

# YOLO model class indices must start at 1
# key: input dataset format class index
# value: YOLO dataset format class index
yolo_classes = {
    11: 1,
    14: 0
}

# Fraction of train data that is used as validation data instead
train_val_split = 0.2

rng = np.random.default_rng(12345)


def process_class_train(class_id, in_dir, out_train_image_dir, out_train_label_dir, out_val_image_dir,
                        out_val_label_dir):
    global num_processed_train_img

    print(f"processing train data for class {class_id}...")
    class_dir_name = str(class_id).zfill(5)

    # open annotation file
    annotations_df = pd.read_csv(in_dir + os.sep + class_dir_name +
                                 os.sep + f"GT-{class_dir_name}.csv", delimiter=';', header=0)

    # iterate over class images
    for index, row in annotations_df.iterrows():
        file_name = row['Filename']

        # Randomize whether it will be a test or val image
        random = rng.random()
        out_image_dir = out_train_image_dir
        out_label_dir = out_train_label_dir
        if random < train_val_split:
            out_image_dir = out_val_image_dir
            out_label_dir = out_val_label_dir

        # Convert image
        in_file_path = in_dir + os.sep + class_dir_name + os.sep + file_name
        out_file_path = out_image_dir + os.sep + \
                        f"{str(num_processed_train_img).zfill(7)}.jpg"

        print(f"Saving to {out_file_path}...")

        img = Image.open(in_file_path)
        img.save(out_file_path)

        # Convert labels
        out_file_path = out_label_dir + os.sep + \
                        f"{str(num_processed_train_img).zfill(7)}.txt"

        # Top left corner
        bounding_box_x1 = int(row['Roi.X1'])
        bounding_box_y1 = int(row['Roi.Y1'])

        # Bottom right corner
        bounding_box_x2 = int(row['Roi.X2'])
        bounding_box_y2 = int(row['Roi.Y2'])

        width = ((bounding_box_x2 - bounding_box_x1) / img.size[0])
        height = ((bounding_box_y2 - bounding_box_y1) / img.size[1])
        x_center = (bounding_box_x1 + (bounding_box_x2 -
                                       bounding_box_x1) / 2) / img.size[0]
        y_center = (bounding_box_y1 + (bounding_box_y2 -
                                       bounding_box_y1) / 2) / img.size[1]

        file = open(out_file_path, 'w')
        file.write(
            f"{yolo_classes[class_id]} {x_center} {y_center} {width} {height}")
        file.close()

        # Other stuff
        num_processed_train_img += 1


def process_class_test(class_index, in_dir, out_image_dir, out_label_dir):
    global num_processed_test_img
    print(f"processing test data for class {class_index}...")

    # open annotation file
    annotations_df = pd.read_csv(
        in_dir + os.sep + 'GT-final_test.csv', delimiter=';', header=0)

    # iterate over images
    for index, row in annotations_df.iterrows():
        if row['ClassId'] != class_index:
            continue

        file_name = row['Filename']

        # Convert image
        in_file_path = in_dir + os.sep + file_name
        out_file_path = out_image_dir + os.sep + \
                        f"{str(num_processed_test_img).zfill(7)}.jpg"

        print(f"Saving to {out_file_path}...")

        img = Image.open(in_file_path)
        img.save(out_file_path)

        # Convert labels
        out_file_path = out_label_dir + os.sep + \
                        f"{str(num_processed_test_img).zfill(7)}.txt"

        # Top left corner
        bounding_box_x1 = int(row['Roi.X1'])
        bounding_box_y1 = int(row['Roi.Y1'])

        # Bottom right corner
        bounding_box_x2 = int(row['Roi.X2'])
        bounding_box_y2 = int(row['Roi.Y2'])

        width = (bounding_box_x2 - bounding_box_x1) / img.size[0]
        height = (bounding_box_y2 - bounding_box_y1) / img.size[1]
        x_center = (bounding_box_x2 - (bounding_box_x2 - bounding_box_x1) / 2) / img.size[0]
        y_center = (bounding_box_y2 - (bounding_box_y2 - bounding_box_y1) / 2) / img.size[1]

        file = open(out_file_path, 'w')
        file.write(
            f"{yolo_classes[class_index]} {x_center} {y_center} {width} {height}")
        file.close()

        # Other stuff
        num_processed_test_img += 1


def process_class(class_index):
    # Training data
    process_class_train(class_index, IN_TRAIN_DIR,
                        OUT_IMAGE_DIR + os.sep + OUT_TRAIN_DIR_NAME,
                        OUT_LABEL_DIR + os.sep + OUT_TRAIN_DIR_NAME,
                        OUT_IMAGE_DIR + os.sep + OUT_VAL_DIR_NAME,
                        OUT_LABEL_DIR + os.sep + OUT_VAL_DIR_NAME
                        )

    # Test data
    process_class_test(class_index, IN_TEST_DIR,
                       OUT_IMAGE_DIR + os.sep + OUT_TEST_DIR_NAME,
                       OUT_LABEL_DIR + os.sep + OUT_TEST_DIR_NAME
                       )


def main():
    print("Creating processed dataset directories...")

    if os.path.exists(OUT_DIR):
        print(f"ATTENTION: output directory '{OUT_DIR}' already exists!")
        print(
            "Irreversibly DELETE this directory and process the source dataset again? y/n [n]")
        if input() != 'y':
            print("Aborting.")
            return

        shutil.rmtree(OUT_DIR)

    os.mkdir(OUT_DIR)
    os.mkdir(OUT_IMAGE_DIR)
    os.mkdir(OUT_LABEL_DIR)

    os.mkdir(OUT_IMAGE_DIR + os.sep + OUT_TRAIN_DIR_NAME)
    os.mkdir(OUT_IMAGE_DIR + os.sep + OUT_VAL_DIR_NAME)
    os.mkdir(OUT_IMAGE_DIR + os.sep + OUT_TEST_DIR_NAME)

    os.mkdir(OUT_LABEL_DIR + os.sep + OUT_TRAIN_DIR_NAME)
    os.mkdir(OUT_LABEL_DIR + os.sep + OUT_VAL_DIR_NAME)
    os.mkdir(OUT_LABEL_DIR + os.sep + OUT_TEST_DIR_NAME)

    process_class(11)  # this junction priority sign
    process_class(14)  # stop sign


if __name__ == "__main__":
    main()
