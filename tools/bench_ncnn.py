import os
import json
import math
import cv2
import numpy as np
import torch
import ncnn
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nanodet.util import cfg, load_config
from nanodet.model.arch import build_model

CONFIG_PATH = '../config/student_deploy.yml'
MODEL_PARAM = '../student_int8.param'  
MODEL_BIN = '../student_int8.bin'
INPUT_SIZE = 320  
VAL_IMG_DIR = '../coco/val2017'  
VAL_ANN_JSON = '../coco/annotations/val_car.json'  


def main():
    load_config(cfg, CONFIG_PATH)
    dummy_model = build_model(cfg.model)
    head = dummy_model.head
    net = ncnn.Net()
    net.load_param(MODEL_PARAM)
    net.load_model(MODEL_BIN)
    coco_gt = COCO(VAL_ANN_JSON)
    image_ids = coco_gt.getImgIds()
    results = []
    for img_id in tqdm(image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMG_DIR, img_info['file_name'])
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue

        h_raw, w_raw = img_raw.shape[:2]
        mat_in = ncnn.Mat.from_pixels_resize(
            img_raw, ncnn.Mat.PixelType.PIXEL_BGR, w_raw, h_raw, INPUT_SIZE, INPUT_SIZE
        )
        mean = [103.53, 116.28, 123.675]
        norm = [0.017429, 0.017507, 0.017124]
        mat_in.substract_mean_normalize(mean, norm)
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")

        if ret != 0:
            print("Failed to extract out0")
            return

        data = np.array(mat_out)
        if len(data.shape) == 1:
            data = data.reshape((data.shape[0] // 33, 33))

        preds_tensor = torch.from_numpy(data).unsqueeze(0).float()
        with torch.no_grad():
            dets = head.post_process(preds_tensor,
                                     meta={
                                         'img': torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE),
                                         'warp_matrix': [np.eye(3)],
                                         'img_info': {
                                             'height': [INPUT_SIZE],
                                             'width': [INPUT_SIZE],
                                             'id': [img_id]
                                         }
                                     })

        scale_x = w_raw / INPUT_SIZE
        scale_y = h_raw / INPUT_SIZE
        img_dets = dets[img_id]
        for label, bboxes in img_dets.items():
            category_id = 0
            for box in bboxes:
                score = float(box[4])
                if score > 0.05:
                    x1 = float(box[0] * scale_x)
                    y1 = float(box[1] * scale_y)
                    x2 = float(box[2] * scale_x)
                    y2 = float(box[3] * scale_y)

                    w = x2 - x1
                    h = y2 - y1

                    results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                        "score": round(score, 5)
                    })

    res_file = f"{MODEL_PARAM}.json"
    with open(res_file, 'w') as f:
        json.dump(results, f)

    if len(results) == 0:
        print("\nmAP = 0!")
        return

    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    main()
