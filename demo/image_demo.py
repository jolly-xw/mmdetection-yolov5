# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img', default="demo/yolov5_onnx_test/images/000005.jpg", help='Image file')
    parser.add_argument('--config',
                        default="configs/yolov5/train/pretrained/yolov5s/yolov5s_csp_mstrain_640_1x8_300e_voc.py",
                        help='Config file')
    parser.add_argument('--checkpoint',
                        default="finetune_voc_aug/epoch_300.pth",
                        help='Checkpoint file')
    parser.add_argument('--device', default='cuda:2', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.25, help='bbox score threshold')
    parser.add_argument('--async-test', action='store_true', help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
