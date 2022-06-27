import torch
import torchvision
import cv2
import os
import detectron2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
import sys


alllimgs = ["wires.jpeg","rebar.jpg","PVC.jpg","debris.png","brick.jpg"]

modelpre = "mask_rcnn_R_101_C4_3x"
folder_dir = "./"
for image_name in alllimgs:
    output_name = "./detections/"+image_name.split(".")[0]+"_detected."+image_name.split(".")[1]
    print(output_name)
    im = cv2.imread(os.path.join(
                        folder_dir, image_name))
    print(os.path.join(
                        folder_dir, image_name))
    # Detection Classes
    classes = ['brick','cementitious_debris','PVC','rebar','wires']
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/"+modelpre+".yaml"))
    cfg.DATASETS.TRAIN = ("category_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/"+modelpre+".yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # Change this according to your classes
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(
        cfg.OUTPUT_DIR, modelpre+".pth")
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (0.5)
    cfg.DATASETS.TEST = ("crack_test", )
    predictor = DefaultPredictor(cfg)
    # Detect
    outputs = predictor(im)
    MetadataCatalog.get("category_train").set(
                        thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get(
                        "category_train")

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(
        im[:, :, ::-1], metadata=microcontroller_metadata, scale=1.2)
    out = v.draw_instance_predictions(
        outputs["instances"].to("cpu"))

    cv2.imwrite(output_name, out.get_image()
                                    [:, :, ::-1])  # mask