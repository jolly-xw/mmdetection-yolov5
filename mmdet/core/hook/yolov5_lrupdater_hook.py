from pickletools import optimize
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import LrUpdaterHook
import numpy as np


@HOOKS.register_module()
class LambdaLrUpdaterHook(LrUpdaterHook):

    def __init__(self, lrf=0.01, **kwargs):
        self.lrf = lrf
        super(LambdaLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters

        def lf(x): return (1 - x / max_progress) * (1.0 - self.lrf) + self.lrf
        lambda_lr = lf(progress) * base_lr
        return lambda_lr


@HOOKS.register_module()
class YOLOV5LrUpdaterHook(LambdaLrUpdaterHook):
    def __init__(self, lr0, warmup_bias_lr, momentum, warmup_momentum, **kwargs):
        self.init_lr = lr0
        self.warmup_bias_lr = warmup_bias_lr
        self.momentum = momentum
        self.warmup_momentum = warmup_momentum
        super(YOLOV5LrUpdaterHook, self).__init__(**kwargs)

    def get_warmup_lr(self, runner, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            """
            Args:
                regular(list):
                cur_iters(int):
            """
            # interp warmup scheme
            warmup_lr = []
            for i, _lr in enumerate(regular_lr):
                warmup_lr.append(np.interp(cur_iters, [0, self.warmup_iters], [0.0 if runner.optimizer.param_groups[i].get('flag')
                                                                              in ['g0', 'g1'] else self.warmup_bias_lr, _lr]))
                # warmup_lr.append(np.interp(cur_iters,[0,runner.max_iters],[self.warmup_bias_lr,_lr*self.init_lr])) # 查看第三条曲线
            # warmup momentum
            momentun_w = np.interp(cur_iters, [0, self.warmup_iters], [
                                   self.warmup_momentum, self.momentum])
            for g in runner.optimizer.param_groups:
                # get momentum
                g['momentum'] = momentun_w
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, base_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, base_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(runner, cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(runner, cur_iter)
                self._set_lr(runner, warmup_lr)
