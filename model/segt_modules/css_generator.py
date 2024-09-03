import math
import torch
from torch import nn
import cv2

from . import loss
import torchvision.transforms.functional as F
from .cldice import soft_dice_cldice 

def unscale(img, min_max=(0, 1)):

    img = (img-min_max[0])/(min_max[1] - min_max[0]) 
    return img


class CssGenerator(nn.Module):
    def __init__(
        self, segment_fn,
        image_size,
        channels=3,
        loss_type='l1',

        opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.segment_fn = segment_fn
        self.loss_type = loss_type
        self.opt=opt

#############defining loss functions###########
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()
        self.loss_nce = []
        for i in range(3):
            self.loss_nce.append(loss.MaskPatchNCELoss().to(device))
        
        self.loss_gan = loss.GANLoss('lsgan').to(device)
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_cldice=soft_dice_cldice().to(device)
        self.loss_bce=nn.BCELoss().to(device)
        



    def p_sample_segment(self, x_in, opt):
        x_start_ = x_in['A']
        segm_V = torch.zeros_like(x_start_)
        dsize = x_start_.shape[-1]

        if opt['phase'] != 'train':
            if 'STARE' in opt['datasets']['test']['dataroot'] or 'DRIVE' in opt['datasets']['test']['dataroot']:
                for opt1 in range(0,dsize,256):
                    for opt2 in range(0,dsize,256):
                        x_start = x_start_[:, :, opt1:opt1+256, opt2:opt2+256]
                        segm_V[:, :, opt1:opt1+256, opt2:opt2+256] = self.segment_fn(torch.cat([x_start, x_start], dim=1))

                return segm_V
            
        for opt1 in range(2):
            for opt2 in range(2):
                x_start = x_start_[:, :, opt1::2, opt2::2]
                segm_V[:, :, opt1::2, opt2::2] = self.segment_fn(torch.cat([x_start, x_start], dim=1))
        return segm_V


    @torch.no_grad()
    def segment(self, x_in, opt):
        return self.p_sample_segment(x_in, opt)


    def p_losses(self, x_in, noise=None):
        
        a_start = x_in['A']
        x_img=a_start.cpu().numpy().squeeze().squeeze()
        gaussian_3 = F.to_tensor(cv2.GaussianBlur(x_img, (0, 0), 5.0)).unsqueeze(0).cuda()
        unsharp_image = F.to_tensor(cv2.addWeighted(x_img, 2.0, gaussian_3.cpu().numpy().squeeze().squeeze(), -1.0, 0)).unsqueeze(0).cuda()

        device = a_start.device
        [b, c, h, w] = a_start.shape


        #### A path ####
        mask_V = self.segment_fn(torch.cat([a_start, unsharp_image], dim=1))
        fractal = torch.eye(2,device=device)[:, torch.clamp_min(x_in['F'][:, 0], 0).type(torch.long)].transpose(0, 1)
        synt_A = self.segment_fn(torch.cat([ a_start,a_start], dim=1), fractal.to(device))

        #### Cycle path ####
        mask_F = self.segment_fn(torch.cat([synt_A, synt_A], dim=1))
        mask_V1 = torch.eye(2,device=device)[:, torch.clamp_min(mask_V[:, 0], 0).type(torch.long)].transpose(0, 1)
        l_recon=self.loss_func(self.segment_fn(torch.cat([a_start, unsharp_image], dim=1),mask_V1),a_start) 
        l_recon=l_recon.sum() / int(b * c * h * w)


        new_X=unscale(x_in['F'],min_max=(-1,1))
        new_X=(new_X > 0.6).float() * 1

        l_l1=self.loss_l1(mask_F,x_in['F'])
        l_cldice=self.loss_cldice(unscale(mask_F,min_max=(-1,1)),new_X)
        l_bce=self.loss_bce(unscale(mask_F,min_max=(-1,1)),new_X)

        return [ gaussian_3, gaussian_3, gaussian_3, mask_V, synt_A, mask_F], [l_recon, l_l1, l_bce, l_cldice] #
        
    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
    


