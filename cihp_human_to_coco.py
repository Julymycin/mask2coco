import argparse
import datetime
import json
import os
from PIL import Image
import PIL
import numpy as np
import cv2

import pycococreatortools

import warnings
warnings.filterwarnings("error", category=UserWarning)

import imageio.plugins.gdal as gdal


def get_arguments():
    parser = argparse.ArgumentParser(
        description="transform mask annotation to coco annotation")
    parser.add_argument("--dataset",
                        type=str,
                        default='CIHP_human',
                        help="name of dataset (CIHP, MHPv2 or VIP)")
    parser.add_argument(
        "--json_save_dir",
        type=str,
        default='/home/qiu/Downloads/datasets/instance-level_human_parsing',
        help="path to save coco-style annotation json file")
    parser.add_argument("--use_val",
                        type=bool,
                        default=False,
                        help="use train+val set for finetuning or not")
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default=
        '/home/qiu/Downloads/datasets/instance-level_human_parsing/Training/Images',
        help="train image path")
    parser.add_argument(
        "--train_anno_dir",
        type=str,
        default=
        '/home/qiu/Downloads/datasets/instance-level_human_parsing/Training/Human_ids',
        help="train human mask path")
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default=
        '/home/qiu/Downloads/datasets/instance-level_human_parsing/Validation/Images',
        help="val image path")
    parser.add_argument(
        "--val_anno_dir",
        type=str,
        default=
        '/home/qiu/Downloads/datasets/instance-level_human_parsing/Validation/Human_ids',
        help="val human mask path")
    return parser.parse_args()


def main(args):
    INFO = {
        "description": args.dataset + " Dataset",
        "url": "",
        "version": "",
        "year": 2020,
        "contributor": "qiu",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [{"id": 1, "name": "", "url": ""}]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
        },
    ]

    # coco_output = {
    #     "info": INFO,
    #     "licenses": LICENSES,
    #     "categories": CATEGORIES,
    #     "images": [],
    #     "annotations": []
    # }
    # image_id = 1
    # segmentation_id = 1
    # train_imgs = os.listdir(args.train_img_dir)
    # for idx, image_name in enumerate(train_imgs):
    #     if image_name.endswith(('jpg', 'png')):
    #         # print(image_name+'\n')
    #         try:
    #             image = Image.open(os.path.join(args.train_img_dir,
    #                                             image_name))
    #         except:
    #             print('corrupt img', image_name)
    #         image_info = pycococreatortools.create_image_info(
    #             image_id, image_name, image.size)
    #         coco_output["images"].append(image_info)
    #         mask_name_prefix = image_name.split('.')[0]
    #         mask_path = os.path.join(args.train_anno_dir,
    #                                  mask_name_prefix + '.png')
    #         mask = np.asarray(Image.open(mask_path))
    #         if len(mask.shape) == 3:
    #             mask = mask[:, :, 0]
    #         gt_labels = np.unique(mask)
    #         for k in range(1, len(gt_labels)):
    #             category_info = {'id': 1, 'is_crowd': 0}
    #             binary_mask = np.uint8(mask == k)
    #             annotation_info = pycococreatortools.create_annotation_info(
    #                 segmentation_id,
    #                 image_id,
    #                 category_info,
    #                 binary_mask,
    #                 image.size,
    #                 tolerance=0)
    #             if annotation_info is not None:
    #                 coco_output["annotations"].append(annotation_info)

    #             segmentation_id += 1
    #         image_id += 1
    #     if image_id % 100 == 0:
    #         print(image_id)
    # if not os.path.exists(args.json_save_dir):
    #     os.makedirs(args.json_save_dir)

    # if not args.use_val:
    #     with open('{}/{}_train.json'.format(args.json_save_dir, args.dataset),
    #               'w') as output_json_file:
    #         json.dump(coco_output, output_json_file)

    # val
    coco_output_val = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id_val = 1
    segmentation_id_val = 1
    val_imgs = os.listdir(args.val_img_dir)
    for idx, image_name in enumerate(val_imgs):
        if image_name.endswith(('jpg', 'png')):
            # print(image_name+'\n')
            try:
                image = Image.open(os.path.join(args.val_img_dir, image_name))
            except:
                print('corrupt img', image_name)
            image_info = pycococreatortools.create_image_info(
                image_id_val, image_name, image.size)
            coco_output_val["images"].append(image_info)
            mask_name_prefix = image_name.split('.')[0]
            mask_path = os.path.join(args.val_anno_dir,
                                     mask_name_prefix + '.png')
            mask = np.asarray(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            gt_labels = np.unique(mask)
            for k in range(1, len(gt_labels)):
                category_info = {'id': 1, 'is_crowd': 0}
                binary_mask = np.uint8(mask == k)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id_val,
                    image_id_val,
                    category_info,
                    binary_mask,
                    image.size,
                    tolerance=0)
                if annotation_info is not None:
                    coco_output_val["annotations"].append(annotation_info)

                segmentation_id_val += 1
            image_id_val += 1
        if image_id_val % 100 == 0:
            print(image_id_val)
    if not os.path.exists(args.json_save_dir):
        os.makedirs(args.json_save_dir)

    if not args.use_val:
        with open('{}/{}_val.json'.format(args.json_save_dir, args.dataset),
                  'w') as output_json_file_val:
            json.dump(coco_output_val, output_json_file_val)


def test(args):
    train_imgs = os.listdir(args.train_img_dir)
    for idx, image_name in enumerate(train_imgs):
        if image_name.endswith(('jpg', 'png')):
            try:
                image = Image.open(os.path.join(args.train_img_dir, image_name))
                image.getexif()
            except:
                print('train corrupt img', image_name)
            # PIL.Image(os.path.join(args.train_img_dir, image_name)).getexif()

    val_imgs = os.listdir(args.val_img_dir)
    for idx, image_name in enumerate(val_imgs):
        if image_name.endswith(('jpg', 'png')):
            try:
                image = Image.open(os.path.join(args.val_img_dir, image_name))
                image.getexif()
                # PIL.Image(os.path.join(args.train_img_dir, image_name)).getexif()
            except:
                print('val corrupt img', image_name)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
    # test(args)
