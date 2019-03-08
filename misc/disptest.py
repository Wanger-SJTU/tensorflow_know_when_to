
from config.config import Config

from utils.dataprovider import DataProvider

from utils.metrics.pycocoevalcap.eval import COCOEvalCap

print("Evaluating the model ...")
config = Config()
config.phase = 'test'
    
test_data = DataProvider(config)
test_gt_coco = test_data.returncoco()
# Evaluate these captions
test_result_coco = test_gt_coco.loadRes(config.test_result_file)
scorer = COCOEvalCap(test_gt_coco, test_result_coco)
scorer.evaluate()
