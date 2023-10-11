import argparse, os, sys, json
import matplotlib.pyplot as plt
import torch
# import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def argumenterParser():
    parser = argparse.ArgumentParser(description='Train and test Detectron2')
    parser.add_argument('--mode', dest='mode',
                      help='choose to train or to test',
                      default='train', type=str)
    parser.add_argument('--lr', dest='learning_rate', default= '0.0125', type=float)
    parser.add_argument('--it', dest='max_iteration', default= '1500', type=int)
    parser.add_argument('--workers', dest='number_of_workers', default= '2', type=int)
    parser.add_argument('--ims_per_batch', dest='ims_per_batch', default= '4', type=int)
    parser.add_argument('--eval_period', dest='evaluation_period', default= '500', type=int)
    parser.add_argument('--batch_size', dest='batch_size', default= '256', type=int)
    parser.add_argument('--output_dir', dest='output_dir', default= '../output', type=str)
    parser.add_argument('--inference', dest='inference', default=False, type=bool)
    parser.add_argument('--visualize_loss', dest='visualize_loss', default=False, type=bool)
    args = parser.parse_args()
    return args

args = argumenterParser()

def registerInstances():
    register_coco_instances("pklot_test", {}, "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/valid/_annotations.coco.json", "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/valid")
    register_coco_instances("pklot_train", {}, "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/train/_annotations.coco.json", "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/train")
    register_coco_instances("pklot_valid", {}, "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/test/_annotations.coco.json", "/home/emiltovborg-jensen/Desktop/Development/Projects/PKLot Object Detection/Data/test")
    
print("Registering instances...")
registerInstances()

print("Creating Config...")
def config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pklot_train",)
    cfg.DATASETS.TEST = ("pklot_valid",)
    cfg.DATALOADER.NUM_WORKERS = args.number_of_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.learning_rate # pick a good LR
    cfg.SOLVER.MAX_ITER = args.max_iteration    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    cfg.TEST.EVAL_PERIOD = args.evaluation_period
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = args.output_dir
    return cfg

import random
import cv2

def visualize_data():
    dataset_dicts = DatasetCatalog.get("pklot_train")
    for d in random.sample(dataset_dicts, 3):
        print(d["annotations"])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("pklot_train"), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('custom', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)

def train():
    cfg = config()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def test():
    cfg = config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    test_set = DatasetCatalog.get("pklot_test")
    for d in random.sample(test_set, 7):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                    metadata=MetadataCatalog.get("pklot_test"), 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        cv2.imshow('custom', img)
        cv2.waitKey(0)
        os.makedirs(f"{cfg.OUTPUT_DIR}/visualizations", exist_ok=True)
        plt.imsave(os.path.join(cfg.OUTPUT_DIR, 'visualizations', f"image_{d['file_name'].split('/')[-1]}"), img)
        print(f"IMG saved to path: {os.path.join(cfg.OUTPUT_DIR, 'visualizations')}")


def inference():    
    cfg = config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("pklot_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pklot_valid")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    return "Inference done!"

if __name__ == '__main__':
    args = argumenterParser()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
        if args.inference:
            inference()