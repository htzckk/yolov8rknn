import time

import cv2
from rknnlite.api import RKNNLite
import platform

from my_data_process import *



class RKNN_model():
    def __init__(self, model_path, target=None, device_id=None) -> None:
        rknn = RKNNLite()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target == None:
            ret = rknn.init_runtime()
        else:
            ret = rknn.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    def infer(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]
        s=time.time()
        result = self.rknn.inference(inputs=inputs)
        e=time.time()
        print('推理时间{}'.format(e-s))

        return result


model_path='./yolov8s_rk3588.rknn'
img_path='./bus.jpg'

model=RKNN_model(model_path=model_path)

img=cv2.imread(img_path)
img=letterbox_r(img)

outputs=model.infer(img)

boxes, classes, scores = post_process(outputs)
draw(img, boxes, scores, classes)

cv2.imwrite('./res_bus.jpg',img)
cv2.destroyAllWindows()

RKNN.release()