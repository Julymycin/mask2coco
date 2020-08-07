import argparse
import datetime
import json
import os
from PIL import Image
import numpy as np
import cv2

import pycococreatortools


def get_arguments():
    parser = argparse.ArgumentParser(
        description="transform mask annotation to coco annotation")
    parser.add_argument("--dataset",
                        type=str,
                        default='MHPv2',
                        help="name of dataset (CIHP, MHPv2 or VIP)")
    parser.add_argument("--json_save_dir",
                        type=str,
                        default='/home/qiu/Downloads/datasets/LV-MHP-v2',
                        help="path to save coco-style annotation json file")
    parser.add_argument("--use_val",
                        type=bool,
                        default=False,
                        help="use train+val set for finetuning or not")
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default='/home/qiu/Downloads/datasets/LV-MHP-v2/train/images',
        help="train image path")
    parser.add_argument(
        "--train_anno_dir",
        type=str,
        default='/home/qiu/Downloads/datasets/LV-MHP-v2/train/parsing_annos',
        help="train human mask path")
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default='/home/qiu/Downloads/datasets/LV-MHP-v2/val/images',
        help="val image path")
    parser.add_argument(
        "--val_anno_dir",
        type=str,
        default='/home/qiu/Downloads/datasets/LV-MHP-v2/val/parsing_annos',
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
            'name': 'foot',
            'supercategory': 'person',
        },
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    mhp2_foot_labels=[20,21,22,23,24,25,26,27,28,29,32,33]

    # image_id = 1
    # segmentation_id = 1
    # train_imgs = os.listdir(args.train_img_dir)
    # for idx, image_name in enumerate(train_imgs):
    #     if image_name.endswith(('jpg', 'png')):
    #         # image = Image.open(os.path.join(args.train_img_dir, image_name))
    #         image=cv2.imread(os.path.join(args.train_img_dir, image_name))
    #         image_info = pycococreatortools.create_image_info(
    #             image_id, image_name, image.shape)
    #         coco_output["images"].append(image_info)
    #         mask_name_prefix = image_name.split('.')[0]
    #         temp_i = 0
    #         # mask_temp = np.ones((image.size[1], image.size[0]),dtype=np.uint8)
    #         mask_temp = np.ones((image.shape[0],image.shape[1]),dtype=np.uint8)
    #         while (True):
    #             mask_path = os.path.join(
    #                 args.train_anno_dir,
    #                 mask_name_prefix + '_02_0' + str(temp_i+1) + '.png')
    #             if os.path.exists(mask_path):
    #                 mask = np.asarray(Image.open(mask_path))
    #                 # mask=cv2.imread(mask_path)
    #                 if len(mask.shape)==3:
    #                     mask=mask[:,:,0]
    #                 mask_temp = np.maximum(mask_temp,mask)
    #                 temp_i += 1
    #             else:
    #                 break
    #         mask_temp=np.where((mask_temp>=20)&(mask_temp<=33)&(mask_temp!=30)&(mask_temp!=31),1,0)
    #         # cv2.imshow("ss",mask_temp*255.0)
    #         # cv2.waitKey(-1)
    #         gt_labels = np.unique(mask_temp)
    #         for i in range(1, len(gt_labels)):
    #             category_info = {'id': 1, 'is_crowd': 0}
    #             binary_mask = np.uint8(mask_temp)
    #             annotation_info = pycococreatortools.create_annotation_info(
    #                 segmentation_id,
    #                 image_id,
    #                 category_info,
    #                 binary_mask,
    #                 binary_mask.shape,
    #                 tolerance=10)
    #             if annotation_info is not None:
    #                 coco_output["annotations"].append(annotation_info)

    #             segmentation_id += 1
    #         image_id += 1
    # if not os.path.exists(args.json_save_dir):
    #     os.makedirs(args.json_save_dir)

    # if not args.use_val:
    #     with open(
    #             '{}/{}_train.json'.format(args.json_save_dir, args.dataset),
    #             'w') as output_json_file:
    #         json.dump(coco_output, output_json_file)
    

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
            # image = Image.open(os.path.join(args.val_img_dir, image_name))
            image=cv2.imread(os.path.join(args.val_img_dir, image_name))
            image_info = pycococreatortools.create_image_info(
                image_id_val, image_name, image.shape)
            coco_output_val["images"].append(image_info)
            mask_name_prefix = image_name.split('.')[0]
            temp_i = 0
            # mask_temp = np.ones((image.size[1], image.size[0]))
            mask_temp = np.ones((image.shape[0],image.shape[1]),dtype=np.uint8)
            while (True):
                mask_path = os.path.join(
                    args.val_anno_dir,
                    mask_name_prefix + '_02_0' + str(temp_i+1) + '.png')
                if os.path.exists(mask_path):
                    mask = np.asarray(Image.open(mask_path))
                    # mask=cv2.imread(mask_path)
                    if len(mask.shape)==3:
                        mask=mask[:,:,0]
                    mask_temp = np.maximum(mask_temp,mask)
                    temp_i += 1
                else:
                    break
            mask_temp=np.where((mask_temp>=20)&(mask_temp<=33)&(mask_temp!=30)&(mask_temp!=31),1,0)
            gt_labels = np.unique(mask_temp)
            for i in range(1, len(gt_labels)):
                category_info = {'id': 1, 'is_crowd': 0}
                binary_mask = np.uint8(mask_temp)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id_val,
                    image_id_val,
                    category_info,
                    binary_mask,
                    binary_mask.shape,
                    tolerance=10)
                if annotation_info is not None:
                    coco_output_val["annotations"].append(annotation_info)

                segmentation_id_val += 1
            image_id_val += 1

    with open('{}/{}_val.json'.format(args.json_save_dir, args.dataset), 'w') as output_json_file_val:
        json.dump(coco_output_val, output_json_file_val)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
