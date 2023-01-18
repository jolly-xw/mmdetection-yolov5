import torch
from collections import OrderedDict

if __name__ == '__main__':
    weights_file = '/data/xuew/Xue_Wei/code/MyProject-MMDet/MyProject/yuml/model_tools/pt_model/ddc.pt'
    weight = torch.load(weights_file, map_location=torch.device('cpu'))['model']
    model = weight.model
    state_dict = model.state_dict()

    index = [10, 24]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        num = int(k.split('.')[0])
        if num < index[0]:  # 前9个是骨架
            name = 'backbone.backbone.' + k
        elif num < index[1]:  # neck
            name = 'bbox_head.det.' + str(num - index[0]) + k[2:]
        else:  # head
            name = 'bbox_head.head.' + k[5:]

        if k.find('anchors') >= 0 or k.find('anchor_grid') >= 0:
            continue

        new_state_dict[name] = v
    # information = {"CLASSES": ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #                            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    #                            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    #                            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    #                            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #                            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #                            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #                            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    #                            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #                            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    #                            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    #                            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    #                            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')}
    information = {"CLASSES": ('missile', )}
    # print(new_state_dict.keys())
    data = {"meta": information, "state_dict": new_state_dict}
    torch.save(data, '../mm_model/ori_yolov5_ddc.pth')
    # saved_model = torch.load('../mm_model/ddc_v6_0_mm.pth')

    print('--')
