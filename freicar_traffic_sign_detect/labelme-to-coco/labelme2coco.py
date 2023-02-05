from pathlib import Path
from PIL import Image
import glob
import json
import os
import shutil

import numpy as np

# INPUT_LABEL_DIR = 'freicar-dataset-1-labelme/labelme'

#INPUT_LABEL_DIR = 'freicar-dataset-2-001-300-labelme/labelme'
#OUT_DIR = 'freicar-dataset-2-001-300-coco'

INPUT_LABEL_DIR = 'freicar-dataset-2-301-600-labelme/updated_jsons_2'
OUT_DIR = 'freicar-dataset-2-301-600-coco'

#INPUT_LABEL_DIR = 'freicar-dataset-2-601-9xx-labelme'
#OUT_DIR = 'freicar-dataset-2-601-9xx-coco'

OUT_IMAGE_DIR = OUT_DIR+os.sep+'images'
OUT_LABEL_DIR = OUT_DIR+os.sep+'labels'
OUT_TRAIN_DIR_NAME = 'train_v1'
OUT_VAL_DIR_NAME = 'val_v1'
OUT_TEST_DIR_NAME = 'test_v1'

val_fraction = 0.2
test_fraction = 0.1
# remaining fraction is train fraction

rng = np.random.default_rng(12345)


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

    label_files = glob.glob(os.path.join(INPUT_LABEL_DIR, "*.json"))
    for image_id, label_file_path in enumerate(label_files):
        print("Converting image:", label_file_path)

        with open(label_file_path) as label_file:
            label_file = json.loads(label_file.read())

        image_path = INPUT_LABEL_DIR + os.sep + label_file['imagePath']

        # Convert Windows to linux path separators
        image_path = image_path.replace('\\', os.sep)

        image_file_name = os.path.basename(image_path)
        image_file_name_without_extension = Path(image_path).stem

        img = Image.open(image_path)

        # Convert label file
        coco_bbox_file_lines = []
        for shape in label_file['shapes']:
            class_id = int(shape['label'])

            if shape['shape_type'] == 'rectangle':
                assert len(shape['points']) == 2

                # Get given data. These are two opposite corners of the bounding box.
                # It is unknown which corners these points belong to.
                x_a = shape['points'][0][0]
                y_a = shape['points'][0][1]
                x_b = shape['points'][1][0]
                y_b = shape['points'][1][1]

                # Make x_1 and y_1 the smaller coordinate components and x_2 and y_2 the larger components
                if x_a < x_b:
                    x_1 = x_a
                    x_2 = x_b
                else:
                    x_1 = x_b
                    x_2 = x_a

                if y_a < y_b:
                    y_1 = y_a
                    y_2 = y_b
                else:
                    y_1 = y_b
                    y_2 = y_a

                # Calculate COCO bounding box format
                width = (x_2-x_1) / img.size[0]
                height = (y_2-y_1) / img.size[1]
                x_center = (x_2 - (x_2-x_1) / 2) / img.size[0]
                y_center = (y_2 - (y_2-y_1) / 2) / img.size[1]

                coco_bbox_file_lines.append(
                    f"{class_id} {x_center} {y_center} {width} {height}")

            elif shape['shape_type'] == 'polygon':
                # YOLO only accepts rectangles, convert the labels
                # assert len(shape['points']) == 4
                raise Exception("TODO")

        # Copy image file and write label file
        random = rng.random()
        image_type = 'train'
        if random < val_fraction:
            image_type = 'val'
        elif random >= val_fraction and random < val_fraction+test_fraction:
            image_type = 'test'

        print(f"Image is type {image_type}")

        if image_type == 'train':
            copy_file(image_path, OUT_IMAGE_DIR + os.sep +
                      OUT_TRAIN_DIR_NAME+os.sep+image_file_name)

            with open(OUT_LABEL_DIR + os.sep + OUT_TRAIN_DIR_NAME+os.sep+image_file_name_without_extension+'.txt', 'w') as label_file:
                label_file.write('\n'.join(coco_bbox_file_lines))

        elif image_type == 'val':
            copy_file(image_path, OUT_IMAGE_DIR + os.sep +
                      OUT_VAL_DIR_NAME+os.sep+image_file_name)

            with open(OUT_LABEL_DIR + os.sep + OUT_VAL_DIR_NAME+os.sep+image_file_name_without_extension+'.txt', 'w') as label_file:
                label_file.write('\n'.join(coco_bbox_file_lines))

        elif image_type == 'test':
            copy_file(image_path, OUT_IMAGE_DIR + os.sep +
                      OUT_TEST_DIR_NAME+os.sep+image_file_name)

            with open(OUT_LABEL_DIR + os.sep + OUT_TEST_DIR_NAME+os.sep+image_file_name_without_extension+'.txt', 'w') as label_file:
                label_file.write('\n'.join(coco_bbox_file_lines))

    # with open(out_ann_file, "w") as f:
    #    json.dump(data, f)


def copy_file(source_path, dest_path):
    with open(dest_path, 'wb') as dest_file:
        with open(source_path, 'rb') as source_file:
            dest_file.write(source_file.read())


if __name__ == "__main__":
    main()
