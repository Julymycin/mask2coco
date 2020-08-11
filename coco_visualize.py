import os

from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

# coco = COCO(json_file)
# # catIds = coco.getCatIds(catNms=['person']) # catIds=1 表示人这一类
# catIds = 1
# imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值
# for i in range(len(imgIds)):
#     img = coco.loadImgs(imgIds[i])[0]
#     I = io.imread(dataset_dir + img['file_name'])
#     plt.axis('off')
#     plt.imshow(I)  # 绘制图像，显示交给plt.show()处理
#     annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#     anns = coco.loadAnns(annIds)
#     coco.showAnns(anns)
#     plt.show()  # 显示图像

from detectron2.data.datasets.coco import load_coco_json
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog


def checkout_dataset_annotation(json_file, dataset_dir, name="MHPv1_foot"):
    dataset_dicts = load_coco_json(json_file, dataset_dir, name)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(name),
                                scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == "__main__":
    MHPv1_foot = {
        'name': 'MHPv1_foot',
        'train_json_file':
        '/home/qiu/Downloads/datasets/LV-MHP-v1/MHPv1_foot_train.json',
        'train_data_dir': '/home/qiu/Downloads/datasets/LV-MHP-v1/images',
    }
    CIHP_human = {
        'name': 'CIHP_human',
        'train_json_file':
        '/home/qiu/Downloads/datasets/instance-level_human_parsing/CIHP_human_train.json',
        'train_data_dir': '/home/qiu/Downloads/datasets/instance-level_human_parsing/Training/Images',
        'val_json_file':'/home/qiu/Downloads/datasets/instance-level_human_parsing/CIHP_human_val.json',
        'val_data_dir': '/home/qiu/Downloads/datasets/instance-level_human_parsing/Validation/Images',
    }
    checkout_dataset_annotation(MHPv1_foot['train_json_file'], MHPv1_foot['train_data_dir'], name=MHPv1_foot['name'])
