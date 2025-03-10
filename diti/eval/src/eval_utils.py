#from __future__ import annotations
from typing import Union, Tuple, List

from PIL import Image, ImageDraw, ImageFont
import torch
from eval.src.metric import maximum, MultiTaskMetric
from sklearn.metrics import average_precision_score, recall_score, f1_score
from scipy.stats import pearsonr
import numpy as np
import torch.nn as nn
import math


class MultiTaskLoss(MultiTaskMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn # should be elementwise
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            flattened_y_pred = flattened_y_pred.float()
            flattened_y_true = flattened_y_true.float()
        elif isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):
            flattened_y_true = flattened_y_true.long()
        flattened_loss = self.loss_fn(flattened_y_pred, flattened_y_true)
        return flattened_loss

    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)
    
def eval_multitask(eval_loader, encoder, classifier, device, mean=None, std=None, attr_idx=-1, avg=False, mask=None):
    y_gt = []
    y_pred = []
    encoder.eval()
    classifier.eval()
    for batch in eval_loader:
        imgs = batch["net_input"]["x_0"].to(device)
        labels = batch["net_input"]["label"].to(device)
        y_gt.append(labels.detach().cpu())
        features = encoder(imgs)
        if mean is not None:
            features = (features - mean) / std
        if mask:
            features = mask(features)
        y_pred.append(classifier(features).detach().cpu())
    y_gt = torch.cat(y_gt, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    if attr_idx >= 0:
        return average_precision_score(y_gt[:,attr_idx], y_pred[:,0])   # pred is only 1 dimensional
    ap_list = []
    for i in range(y_gt.shape[1]):
        ap = average_precision_score(y_gt[:,i], y_pred[:,i])
        ap_list.append(ap)
    if avg:
        return sum(ap_list) / len(ap_list)
    else:
        return ap_list

def eval_regression(eval_loader, encoder, classifier, device):
    y_gt = []
    y_pred = []
    encoder.eval()
    classifier.eval()
    for batch in eval_loader:
        imgs = batch[0].to(device)
        labels = batch[2].to(device)
        y_gt.append(labels.detach().cpu())
        y_pred.append(classifier(encoder(imgs)).detach().cpu())
    y_gt = torch.cat(y_gt, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    mse_per_attribute = ((y_gt-y_pred)**2).mean(axis=0)
    pearsonr_list = []
    for i in range(y_gt.shape[1]):
        pearsonr_list.append(pearsonr(y_gt[:,i], y_pred[:,i]).statistic)
    return pearsonr_list, mse_per_attribute

def accumulate(classifier, ema_classifier, decay):
    classifier.eval()
    ema_classifier.eval()
    ema_classifier_parameter = dict(ema_classifier.named_parameters())
    classifier_parameter = dict(classifier.named_parameters())
    for k in ema_classifier_parameter.keys():
        if classifier_parameter[k].requires_grad:
            ema_classifier_parameter[k].data.mul_(decay).add_(classifier_parameter[k].data, alpha=1.0 - decay)

def get_manipulated_images(encoder, decoder, classifier, diffusion, val_loader, mean, std, attr_idx, mask=None, scale=0.3):
    for batch in val_loader:
        x_0 = batch["net_input"]["x_0"].to(diffusion.device)
        gt = batch['gts']
        inferred_x_T = diffusion.representation_learning_ddim_encode(f'ddim500', encoder, decoder, x_0)
        if attr_idx < 0:
            attr_idx = 31   # default
        else:
            attr_idx = 0    # if single attribute, then classifier only has 1 weight
        with torch.no_grad():
            weight = mask(classifier.weight) if mask else classifier.weight
            images = diffusion.manipulation_sample(
                classifier_weight=weight,
                encoder=encoder,
                decoder=decoder,
                x_0=x_0,
                inferred_x_T=inferred_x_T,
                latents_mean=mean.to(diffusion.device),
                latents_std=std.to(diffusion.device),
                class_id=attr_idx,
                scale=scale,
                ddim_style=f'ddim200',
            )
        images = images.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
        images = images.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        break
    return images, gt


class ProMask(nn.Module):
    def __init__(
            self, 
            dims: int, 
            min_pr: float, 
            pr_start: float,
            pr_end: float,
            max_iters: int,
            pr_schedule: bool = True,
            temp: float  = 1., 
            temp_anneal: bool = True,
            constant_init_value = 1.,
        ):
        super().__init__()
        self.dims = dims
        self.scores = nn.Parameter(torch.Tensor(dims))  #Probability
        self.subnet = None                              #Mask
        self.min_pr = min_pr
        self.pr_start = pr_start
        self.pr_end = pr_end
        self.max_iters = max_iters
        self.temp_anneal = temp_anneal
        self.pr_schedule  = pr_schedule

        self.temp = temp
        self.pr = self.min_pr
        self.adjust_promask(0)  # init temp and pr if use scheduler
        if constant_init_value is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * constant_init_value
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def forward(self, x):  # discrete is not used here
        if self.training:                                      #training
            eps = 1e-20
            temp = self.temp
            uniform0 = torch.rand_like(self.scores)
            uniform1 = torch.rand_like(self.scores)
            noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
            self.subnet = torch.sigmoid((torch.log(self.scores + eps) - torch.log(1.0 - self.scores + eps) + noise) * temp)
        return x * self.subnet if self.subnet is not None else x

    def fix_subnet(self,):
        self.subnet = (torch.rand_like(self.scores) < self.scores).float()
        
    def adjust_promask(self, it: int) -> None:
        """During training stage, adjust pruning rate if use temperature annealing or pr scheduler
        Args:
            it (int): current iteration step
        Returns:
            None
        """
        if self.temp_anneal:
            self.temp = 1 / ((1 - 0.03) * (1 - it / self.max_iters) + 0.03)
        if self.pr_schedule:
            ts = int(self.pr_start * self.max_iters)
            te = int(self.pr_end * self.max_iters)
            pr_target = self.min_pr
            if it < ts:
                self.pr = 1.0
            elif it < te:
                self.pr = pr_target + (1.0 - pr_target)*(1-(it-ts)/(te-ts))**3
            else:
                self.pr = pr_target            

    @torch.no_grad()
    def constrain_mask(self):
        v = self.solve_v_total()
        self.scores.sub_(v).clamp_(0, 1)

    @torch.no_grad()
    def mask_sum(self):
        return (torch.rand_like(self.scores) < self.scores).sum()

    def solve_v_total(self):
        k = self.dims * self.pr
        a = 0
        b = self.scores.max()
        def f(v):
            s = (self.scores - v).clamp(0, 1).sum()
            return s - k
        if f(0) < 0:
            return 0
        itr = 0
        while (1):
            itr += 1
            v = (a + b) / 2
            obj = f(v)
            if abs(obj) < 1e-3 or itr > 20:
                break
            if obj < 0:
                b = v
            else:
                a = v
        v = max(0, v)
        return v


class ImageCanvas:
    def __init__(
            self, 
            nrow: Union[float,int], 
            ncol: Union[float,int], 
            image_size: Union[int, Tuple[int, int]], 
            height_first=False
        ):
        """_summary_

        Args:
            nrow (int): _description_
            ncol (int): _description_
            image_size (Union[int, Tuple[int, int]]):  A 2-tuple, containing (width, height) in pixels.
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple):
            assert len(image_size)==2, f"only accept 2 dim image, but got {len(image_size)} dims."
            if height_first:
                image_size = (image_size[1], image_size[0])  # swap if height first like in torch.Tensor
        else:
            raise TypeError(f"only accept int or tuple image_size, but got {type(image_size)}")

        self.image_size = image_size
        self.canvas = Image.new('RGB', (int(ncol*image_size[0]), int(nrow*image_size[1])), color=(255, 255, 255))

    def grid2pixel(self, t: Tuple[Union[float,int], Union[float,int]]) -> Tuple[int, int]:
        return (int(t[0]*self.image_size[0]), int(t[1]*self.image_size[1]))

    def add_image(
        self, 
        image: Union[torch.Tensor, Image.Image, np.ndarray], 
        loc: Tuple[Union[float,int], Union[float,int]]
    ) -> None:
        """
        loc: grid location (origin at the top-left corner of the canvas) to put the top-left corner of the input image
        the first dim is the horizontal loc (x-axis), the second dim is vertical dim (y-axis)
        """
        assert len(loc)==2, f"only accept length-2 tuple, but got {len(loc)}"
        if isinstance(image, torch.Tensor):
            assert len(image.shape)==4
            image = self.Tensor2Image(image)
        elif isinstance(image, np.ndarray):
            assert len(image.shape)==3
            image = self.Array2Image(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise TypeError(f"only accept type [torch.Tensor, PIL.Image, np.ndarray], but got {type(image)}")

        self.canvas.paste(image, self.grid2pixel(loc))
    
    def add_text(
        self,   
        text: str, 
        loc: Tuple[Union[float,int], Union[float,int]]
    ) -> None:
        """
        loc: pixel location (origin at the top-left corner of the canvas) to draw the top-left corner of the text object
        """
        assert len(loc)==2, f"only accept length-2 tuple, but got {len(loc)}"
        draw = ImageDraw.Draw(self.canvas)
        font = ImageFont.truetype("/$SOMEDIR/miscs/Fonts/dejavu-sans/ttf/DejaVuSans.ttf", 10)
        text_color = (0, 0, 0)
        draw.text(self.grid2pixel(loc), text, font=font, fill=text_color)
    
    @staticmethod
    def Tensor2Image(t) -> Image.Image:
        t = t.squeeze(0)
        assert len(t.shape)==3
        t = t.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255)
        t = t.permute(1,2,0).to('cpu', torch.uint8).numpy()
        t = Image.fromarray(t)
        return t

    @staticmethod
    def Array2Image(t) -> Image.Image:
        return Image.fromarray(t)

    def save(self, path: str) -> None:
        self.canvas.save(path)

    def resize_by_scale(self, s: Union[float, int]) -> Image.Image:
        w, h = self.canvas.size
        new_w, new_h = int(s * w), int(s * h)
        return self.canvas.resize((new_w, new_h))

    @staticmethod
    def build_from_list(
        img_txt_list: List[Tuple[Union[torch.Tensor, Image.Image, np.ndarray], str]], 
        image_size: Union[int, Tuple[int, int]],
        num_text_image_rows: int = 1,
        height_first: bool = False,
        text_image_height_ratio: float = 1,
        txt_shift: float = 0.5):#,
    #) -> ImageCanvas:
        """
        Args:
            num_text_image_rows: the number of rows in which an image is paired with text.
            height_first: whether the height dimension comes first in `image_size`.
                Following the convention of PIL.Image, width comes first by default.
            text_image_height_ratio: the proportional height of the text row to the image row.
                text rows and image rows have the same height by default (text_image_height_ratio=1).
            txt_shift: y-axis location of text in its row.
        """

        ncol = len(img_txt_list) // num_text_image_rows
        nrow = (1 + text_image_height_ratio) * num_text_image_rows
        merge = ImageCanvas(nrow=nrow, ncol=ncol, image_size=image_size, height_first=height_first)
        for i, (img, txt) in enumerate(img_txt_list):
            col_id = i % ncol
            row_id = i // ncol * (1 + text_image_height_ratio)
            merge.add_text(txt, (col_id, row_id + txt_shift*text_image_height_ratio))
            merge.add_image(img, (col_id, row_id + text_image_height_ratio))
        return merge
