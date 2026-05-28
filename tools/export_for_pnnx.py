import torch
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight, Logger

CONFIG_PATH = '../config/student.yml'
MODEL_PATH = '../workspace/student_distilled/model_best/student_distilled_prunned.pth'
OUTPUT_PATH = './student_distilled.pt'
INPUT_SHAPE = (320, 320)

def main():
    load_config(cfg, CONFIG_PATH)
    logger = Logger(-1, cfg.save_dir, False)
    model = build_model(cfg.model)
    checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    load_model_weight(model, checkpoint, logger)
    model.eval()
    dummy_input = torch.rand(1, 3, INPUT_SHAPE[0], INPUT_SHAPE[1])
    traced_mod = torch.jit.trace(model, dummy_input)
    traced_mod.save(OUTPUT_PATH)

if __name__ == '__main__':
    main()
