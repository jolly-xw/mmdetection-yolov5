# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import warnings
from functools import partial
from mmdet.apis import inference_detector, init_detector

import numpy as np
import onnx
import torch
from mmcv import Config, DictAction

from mmdet.core.export import build_model_from_cfg, preprocess_example_input
from mmdet.core.export.model_wrappers import ONNXRuntimeDetector


def pytorch2onnx(model,
                 input_img,
                 input_shape,
                 normalize_cfg,
                 output_filedi=None,
                 opset_version=11,
                 show=True,
                 verify=True,
                 test_img=None,
                 do_simplify=False,
                 dynamic_export=None,
                 skip_postprocess=True):
    """
    Convert pth file to onnx file.
    Remove the postprocess function within pth model, e.g., bbox decode, nms, etc.
    The original outputs of cnn from pth and onnx are compared and error will be printed as result.
    Args:
    """

    # output_file = osp.join('work_dirs', output_filedir, 'LabelImgOnnx.onnx')
    output_fil = osp.join(output_filedi, 'pascal_voc.onnx')
    if normalize_cfg is None:
        input_config = {
            'input_shape': input_shape,
            'input_path': input_img,
        }
    else:
        input_config = {
            'input_shape': input_shape,
            'input_path': input_img,
            'normalize_cfg': normalize_cfg
        }

    # prepare input
    one_img, one_meta = preprocess_example_input(input_config)
    img_list, img_meta_list = [one_img], [[one_meta]]

    if skip_postprocess:
        warnings.warn('Not all models support export onnx without post '
                      'process, especially two stage detectors!')
        model.forward = model.forward_dummy
        torch.onnx.export(
            model,
            one_img,
            output_fil,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=show,
            opset_version=opset_version)

        print(f'Successfully exported ONNX model without '
              f'post process: {output_fil}')
        # return

    if verify:
        # check by onnx
        onnx_model = onnx.load(output_fil)
        onnx.checker.check_model(onnx_model)

        # wrap onnx model
        onnx_model = ONNXRuntimeDetector(output_fil, model.CLASSES, 0)
        if dynamic_export:
            # scale up to test dynamic shape
            h, w = [int((_ * 1.5) // 32 * 32) for _ in input_shape[2:]]
            h, w = min(1344, h), min(1344, w)
            input_config['input_shape'] = (1, 3, h, w)

        if test_img is None:
            input_config['input_path'] = input_img

        # prepare input once again
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # get pytorch output
        with torch.no_grad():
            pytorch_results = model.forward_dummy(one_img)

        img_list = [_.cuda().contiguous() for _ in img_list]
        if dynamic_export:
            img_list = img_list + [_.flip(-1).contiguous() for _ in img_list]
            img_meta_list = img_meta_list * 2
        # get onnx output
        onnx_results = onnx_model.forward_test(
            img_list, img_metas=img_meta_list, return_loss=False)
        # visualize predictions
        score_thr = 0.3

        meanse = 0
        maxse = 0
        for i in range(len(pytorch_results[0])):
            errors = np.abs(onnx_results[i] -
                            pytorch_results[0][i].cpu().numpy())
            # mean square error between onnx_results and pytorch_results
            meanse += np.mean(errors)
            # max square error between onnx_results and pytorch_results
            maxse = max(maxse, np.max(errors))

        print(
            f'The final mean square error between onnx and pth is {meanse/len(pytorch_results[0])}, max sqare error is {maxse}.')

    return


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


r"""
Note: The main function of this file is to convert the model file in pth format generated 
by mmdet training into onnx file for deployment and use on the board side.

Args:
    config : The path of a model config file.
    checkpoint : The path of a model checkpoint file.
    --output-file: The path of output ONNX model. If not specified, it will be set to tmp.onnx.
    --input-img: The path of an input image for tracing and conversion. By default, it will be set to tests/data/color.jpg.
    --shape: The height and width of input tensor to the model. If not specified, it will be set to 800 1216.
    --test-img : The path of an image to verify the exported ONNX model. By default, it will be set to None, meaning it will use --input-img for verification.
    --opset-version : The opset version of ONNX. If not specified, it will be set to 11.
    --dynamic-export: Determines whether to export ONNX model with dynamic input and output shapes. If not specified, it will be set to False.
    --show: Determines whether to print the architecture of the exported model and whether to show detection outputs when --verify is set to True. If not specified, it will be set to False.
    --verify: Determines whether to verify the correctness of an exported model. If not specified, it will be set to False.
    --simplify: Determines whether to simplify the exported ONNX model. If not specified, it will be set to False.
    --cfg-options: Override some settings in the used config file, the key-value pair in xxx=yyy format will be merged into config file.
    --skip-postprocess: Determines whether export model without post process. If not specified, it will be set to False. 
    Notice: This is an experimental option. Only work for some single stage models. 
    Users need to implement the post-process by themselves. We do not guarantee the correctness of the exported model
Return:
Onnx Model,The error between pth and onnx model.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument(
        '--config', default='configs/yolov5/train/pretrained/yolov5s/yolov5s_csp_mstrain_640_1x8_300e_voc.py', help='test config file path')
    parser.add_argument(
        '--checkpoint', default='finetune_voc_aug/epoch_300.pth', help='checkpoint file')
    parser.add_argument('--input-img', default='demo/images/voc_1.jpg',
                        type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='Show onnx graph and detection outputs')
    parser.add_argument(
        '--output-fil', default='model_tools/onnx_model/', type=str)
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str,  default='demo/images/voc_1.jpg',help='Images for test')
    parser.add_argument(
        '--dataset',
        type=str,
        default='coco',
        help='Dataset name. This argument is deprecated and will be removed \
        in future releases.')
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[800, 1216],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[123.675, 116.28, 103.53],
        help='mean value used for preprocess input data.This argument \
        is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[58.395, 57.12, 57.375],
        help='variance value used for preprocess input data. '
        'This argument is deprecated and will be removed in future releases.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    parser.add_argument(
        '--skip-postprocess',
        action='store_true',
        default=True,
        help='Whether to export model without post process. Experimental '
        'option. We do not guarantee the correctness of the exported '
        'model.')
    parser.add_argument(
        '--gpu-ids',
        type=str,
        default='0',
        help='ids of gpus to use ')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    # Initialize the pth model and inference the input picture
    model = init_detector(args.config, args.checkpoint, 'cuda:'+args.gpu_ids)
    res = inference_detector(model, args.input_img)
    model.show_result(args.input_img, res, out_file='show_pt.png')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    # build the model and load checkpoint
    model = build_model_from_cfg(args.config, args.checkpoint,
                                 args.cfg_options)

    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    # normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)
    transforms = []
    for pipeline in cfg.test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    if len(norm_config_li) == 0:
        norm_config = None
    else:
        assert len(norm_config_li) == 1, '`norm_config` should only have one'
        norm_config = norm_config_li[0]
    # convert model to onnx file
    dummy_input = torch.randn(1, 3, 640, 640)
    pytorch2onnx(
        model,
        args.input_img,
        # dummy_input,
        input_shape,
        # normalize_cfg,
        norm_config,
        opset_version=args.opset_version,
        show=args.show,
        output_filedi=args.output_fil,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export,
        skip_postprocess=args.skip_postprocess)
