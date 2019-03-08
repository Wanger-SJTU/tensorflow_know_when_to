
from config.config import Config

from utils.dataprovider import DataProvider

from utils.metrics.pycocoevalcap.eval import COCOEvalCap

print("Evaluating the model ...")
config = Config()
config.phase = 'eval'
    
eval_data = DataProvider(config)
eval_gt_coco = eval_data.returncoco()
# Evaluate these captions
eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
scorer.evaluate()
