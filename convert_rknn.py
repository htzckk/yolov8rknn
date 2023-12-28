import os
import sys
import time

import numpy as np
from rknn.api import RKNN
import cv2
from my_data_process import post_process,draw,letterbox_r,box_process

platform='rk3588'
model_path= './simyolov8n.onnx'
do_quant=False
output_model_path='./yolov8s_rk3588.rknn'
img_path='./bus.jpg'
DATASET_PATH=''


if __name__ == '__main__':


    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[
        [255, 255, 255]], target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path,outputs=['/model.22/cv2.2/cv2.2.2/Conv_output_0','/model.22/cv3.2/cv3.2.2/Conv_output_0',
                                                   '/model.22/cv2.1/cv2.1.2/Conv_output_0','/model.22/cv3.1/cv3.1.2/Conv_output_0',
                                                   '/model.22/cv2.0/cv2.0.2/Conv_output_0','/model.22/cv3.0/cv3.0.2/Conv_output_0'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    # ret = rknn.export_rknn(output_model_path)
    # if ret != 0:
    #     print('Export rknn model failed!')
    #     exit(ret)
    # print('done')

    print('--> Init runtime environment')
    # 需要启动RKNN_server,连版模拟
    # ret = rknn.init_runtime('rk3588',device_id='0029e5bc17f4d68e')
    # 默认值为None, 即在PC使用工具时, 模型在模拟器上运行,需要先调用 build 或 hybrid_quantization 接口才可让模型在模拟器上运行。
    ret=rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    img=cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=letterbox_r(img)
    # print(img.shape)

    s=time.time()
    output=rknn.inference([img])
    e=time.time()
    print('推理时间{}'.format(e-s))
    # print(np.array(output).shape)


    boxes, classes, scores = post_process(output)
    draw(img,boxes,scores,classes)

    cv2.imshow('res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release
    rknn.release()