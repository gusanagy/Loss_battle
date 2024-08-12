"""
All loss functions 
"""

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import mean, round, transpose
from typing import List

def list_channel_loss(list_loss: List[str] = None,rank=0):
    if list_loss == None:
        return []
    dict_loss = {
    'angular_color_loss':angular_color_loss().to(rank),
    'light_loss':light_loss().to(rank),
    'dark_channel_loss':DarkChannelLoss().to(rank),
    'lch_channel_loss':LCHChannelLoss().to(rank),
    'lab_channel_loss':LabChannelLoss().to(rank), 
    'yuv_channel_loss':YUVChannelLoss().to(rank),
    'hsv_channel_loss':HSVChannelLoss().to(rank), 
    'ycbcr_channel_loss':YCbCrChannelLoss().to(rank),
    'cieluv_channel_loss':CIELUVChannelLoss().to(rank),
    'yuv420_channel_loss:':YUV420ChannelLoss().to(rank),
    'Histogram_loss':HistogramColorLoss().to(rank)
                }
    return_loss =[]
    for loss in list_loss:
        if loss not in dict_loss.keys():
            raise ValueError(f"Unsupported perceptual model type\nPlease choose from {dict_loss.keys()}")
        return_loss.append(dict_loss[loss])
    return return_loss


def normalize_loss_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        result  = 1-(1/result)

        # Lidar com valores infinitos no tensor
        result = torch.where(torch.isinf(result), torch.tensor(1.0), result)
        result = torch.where(result == -float('inf'), torch.tensor(0.0), result)
        
        
        # # Obter o valor mínimo e máximo do tensor
        # min_val = torch.min(result)
        # max_val = torch.max(result)
        
        # # Normalização para o intervalo [0, 1]
        # if max_val > min_val:
        #     normalized_result = (result - min_val) / (max_val - min_val)
        # else:
        #     normalized_result = torch.zeros_like(result)
        
        # # Garantir que o tensor esteja no intervalo [0, 1]
        # normalized_result = torch.clamp(normalized_result, 0, 1)
        
        return result
    
    return wrapper

"""Angular Color Loss function"""##mudar nome %
class angular_color_loss(nn.Module):
    def __init__(self,id:int = None):
        super(angular_color_loss, self).__init__()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    
    def forward(self, output, gt,mask=None):
        img_ref = F.normalize(output, p = 2, dim = 1)
        ref_p = F.normalize(gt, p = 2, dim = 1)
        if mask!=None:
            img_ref=mask*img_ref
            ref_p*=mask
        loss_cos = 1 - torch.mean(F.cosine_similarity(img_ref, ref_p, dim=1))
        # loss_cos = self.mse(img_ref, ref_p)
        return loss_cos
    
"""Light Loss Function"""
class light_loss(nn.Module):##pesquisar significado das modificacoes
    def __init__(self,id:int = None):
        super(light_loss, self).__init__()
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def forward(self,output,gt,mask=None):
        #output = torch.mean(output, 1, keepdim=True)
        #gt=torch.mean(gt,1,keepdim=True)
        output =output[:, 0:1, :, :] * 0.299 + output[:, 1:2, :, :] * 0.587 + output[:, 2:3, :, :] * 0.114
        gt = gt[:, 0:1, :, :] * 0.299 + gt[:, 1:2, :, :] * 0.587 + gt[:, 2:3, :, :] * 0.114
        if mask != None:
            output*=mask
            gt*=mask
        loss=F.l1_loss(output,gt)
        return loss

"""Dark Channel Loss"""#%
class DarkChannelLoss(nn.Module):
    def __init__(self,id:int = None, patch_size=15):
        super(DarkChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    @normalize_loss_output
    def forward(self, input, target):
        """
        Compute the Dark Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The Dark Channel Loss value.
        """
        def dark_channel(image, patch_size):
            # Compute the dark channel of an image
            min_channel = torch.min(image, dim=1, keepdim=True)[0]
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            dark_channel = F.conv2d(min_channel, kernel, stride=1, padding=patch_size//2)
            return dark_channel
        
        # Compute dark channels
        dark_input = dark_channel(input, self.patch_size)
        dark_target = dark_channel(target, self.patch_size)
        
        # Compute the loss
        loss = F.mse_loss(dark_input, dark_target,reduction='mean')
        return loss

"""LCH Channel Loss"""
class LCHChannelLoss(nn.Module):
    def __init__(self,id:int = None, patch_size=15):
        super(LCHChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def rgb_to_lch(self, rgb):
        """
        Convert RGB to LCH color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The LCH image with shape (N, 3, H, W).
        """
        # Convert RGB to XYZ
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        
        # Normalize XYZ
        x = x / 0.95047
        z = z / 1.08883
        
        # Convert XYZ to LCH
        l = 116 * y.pow(1/3) - 16
        c = torch.sqrt((x - y.pow(1/3)) ** 2 + (y - z) ** 2)
        h = torch.atan2(x - y.pow(1/3), y - z)
        
        return torch.stack([l, c, h], dim=1)
    
    def _lch_channel(self,image, patch_size:int  = 5):
            # Compute the LCH channels of an image
            lch = self.rgb_to_lch(image)
            l = lch[:, 0, :, :]
            c = lch[:, 1, :, :]
            h = lch[:, 2, :, :]
            
            # Apply convolution to compute dark channel (LCH-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            l_channel = F.conv2d(l.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            c_channel = F.conv2d(c.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            h_channel = F.conv2d(h.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            
            return l_channel, c_channel, h_channel

    def forward(self, input, target):
        """
        Compute the LCH Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The LCH Channel Loss value.
        """
        
        
        # Compute LCH channels
        lch_input = self._lch_channel(input, self.patch_size)
        lch_target = self._lch_channel(target, self.patch_size)
        
        # Compute the loss
        l_loss = F.mse_loss(lch_input[0], lch_target[0],reduction='mean')
        c_loss = F.mse_loss(lch_input[1], lch_target[1],reduction='mean')
        h_loss = F.mse_loss(lch_input[2], lch_target[2],reduction='mean')
        
        # Total loss
        loss = l_loss + c_loss + h_loss
        return (1/loss)-1

"""Lab Channel"""#%
class LabChannelLoss(nn.Module):
    def __init__(self,id:int = None, patch_size=15):
        super(LabChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def rgb_to_lab(self, rgb):
        """
        Convert RGB to LAB color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The LAB image with shape (N, 3, H, W).
        """
        # Convert RGB to XYZ
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]
        
        # Linear RGB to XYZ conversion
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0
        
        # Apply transformation matrix
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        
        # Normalize XYZ
        x = x / 0.95047
        z = z / 1.08883
        
        # Convert XYZ to LAB
        def f(t):
            return torch.where(t > 0.008856, t.pow(1/3), 7.787 * t + 16/116)
        
        l = 116 * f(y) - 16
        a = 500 * (f(x) - f(y))
        b = 200 * (f(y) - f(z))
        
        return torch.stack([l, a, b], dim=1)

    def forward(self, input, target):
        """
        Compute the LAB Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The LAB Channel Loss value.
        """
        def lab_channel(image, patch_size):
            # Compute the LAB channels of an image
            lab = self.rgb_to_lab(image)
            l = lab[:, 0, :, :]
            a = lab[:, 1, :, :]
            b = lab[:, 2, :, :]
            
            # Apply convolution to compute channel loss (LAB-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            l_channel = F.conv2d(l.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            a_channel = F.conv2d(a.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            b_channel = F.conv2d(b.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            
            return l_channel, a_channel, b_channel
        
        # Compute LAB channels
        lab_input = lab_channel(input, self.patch_size)
        lab_target = lab_channel(target, self.patch_size)
        
        # Compute the loss
        l_loss = F.mse_loss(lab_input[0], lab_target[0],reduction='mean')
        a_loss = F.mse_loss(lab_input[1], lab_target[1],reduction='mean')
        b_loss = F.mse_loss(lab_input[2], lab_target[2],reduction='mean')
        
        # Total loss
        loss = l_loss + a_loss + b_loss
        return loss

"""YUV Channel Loss"""
class YUVChannelLoss(nn.Module):
    def __init__(self,id: int = None, patch_size=15):
        super(YUVChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def rgb_to_yuv(self, rgb):
        """
        Convert RGB to YUV color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The YUV image with shape (N, 3, H, W).
        """
        # Convert RGB to YUV
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b
        v = 0.615 * r - 0.51499 * g - 0.10001 * b
        
        return torch.stack([y, u, v], dim=1)

    def forward(self, input, target):
        """
        Compute the YUV Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The YUV Channel Loss value.
        """
        def yuv_channel(image, patch_size):
            # Compute the YUV channels of an image
            yuv = self.rgb_to_yuv(image)
            y = yuv[:, 0, :, :]
            u = yuv[:, 1, :, :]
            v = yuv[:, 2, :, :]
            
            # Apply convolution to compute channel loss (YUV-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            y_channel = F.conv2d(y.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            u_channel = F.conv2d(u.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            v_channel = F.conv2d(v.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            
            return y_channel, u_channel, v_channel
        
        # Compute YUV channels
        yuv_input = yuv_channel(input, self.patch_size)
        yuv_target = yuv_channel(target, self.patch_size)
        
        # Compute the loss
        y_loss = F.mse_loss(yuv_input[0], yuv_target[0],reduction='mean')
        u_loss = F.mse_loss(yuv_input[1], yuv_target[1],reduction='mean')
        v_loss = F.mse_loss(yuv_input[2], yuv_target[2],reduction='mean')
        
        # Total loss
        loss = y_loss + u_loss + v_loss
        return loss

"""HSV Channel Loss"""#%
class HSVChannelLoss(nn.Module):
    def __init__(self,id: int=None, patch_size=15):
        super(HSVChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def rgb_to_hsv(self, rgb):
        """
        Convert RGB to HSV color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The HSV image with shape (N, 3, H, W).
        """
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]

        max_val, _ = torch.max(torch.stack([r, g, b], dim=1), dim=1)
        min_val, _ = torch.min(torch.stack([r, g, b], dim=1), dim=1)

        delta = max_val - min_val
        s = torch.where(max_val == 0, torch.tensor(0.0, device=rgb.device), delta / max_val)

        h = torch.where(
            delta == 0,
            torch.tensor(0.0, device=rgb.device),
            torch.where(
                max_val == r,
                (g - b) / delta % 6,
                torch.where(
                    max_val == g,
                    (b - r) / delta + 2,
                    (r - g) / delta + 4
                )
            ) / 6
        )
        h = (h + 1) % 1

        return torch.stack([h, s, max_val], dim=1)
    @normalize_loss_output
    def forward(self, input, target):
        """
        Compute the HSV Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The HSV Channel Loss value.
        """
        def hsv_channel(image, patch_size):
            # Compute the HSV channels of an image
            hsv = self.rgb_to_hsv(image)
            h = hsv[:, 0, :, :]
            s = hsv[:, 1, :, :]
            v = hsv[:, 2, :, :]

            # Apply convolution to compute channel loss (HSV-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            h_channel = F.conv2d(h.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            s_channel = F.conv2d(s.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            v_channel = F.conv2d(v.unsqueeze(1), kernel, stride=1, padding=patch_size//2)

            return h_channel, s_channel, v_channel

        # Compute HSV channels
        hsv_input = hsv_channel(input, self.patch_size)
        hsv_target = hsv_channel(target, self.patch_size)

        # Compute the loss
        h_loss = F.mse_loss(hsv_input[0], hsv_target[0],reduction='mean')
        s_loss = F.mse_loss(hsv_input[1], hsv_target[1],reduction='mean')
        v_loss = F.mse_loss(hsv_input[2], hsv_target[2],reduction='mean')

        # Total loss
        #loss = (1/(h_loss + s_loss + v_loss))-1
        loss = h_loss + s_loss + v_loss

        return loss

"""YcbCr Channel Loss"""
class YCbCrChannelLoss(nn.Module):
    def __init__(self,id:int = None, patch_size=15):
        super(YCbCrChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def rgb_to_ycbcr(self, rgb):
        """
        Convert RGB to YCbCr color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The YCbCr image with shape (N, 3, H, W).
        """
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]

        # YCbCr conversion coefficients
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
        cr = 0.500 * r - 0.460 * g - 0.040 * b + 128
        
        return torch.stack([y, cb, cr], dim=1)

    def forward(self, input, target):
        """
        Compute the YCbCr Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The YCbCr Channel Loss value.
        """
        def ycbcr_channel(image, patch_size):
            # Compute the YCbCr channels of an image
            ycbcr = self.rgb_to_ycbcr(image)
            y = ycbcr[:, 0, :, :]
            cb = ycbcr[:, 1, :, :]
            cr = ycbcr[:, 2, :, :]

            # Apply convolution to compute channel loss (YCbCr-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            y_channel = F.conv2d(y.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            cb_channel = F.conv2d(cb.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            cr_channel = F.conv2d(cr.unsqueeze(1), kernel, stride=1, padding=patch_size//2)

            return y_channel, cb_channel, cr_channel

        # Compute YCbCr channels
        ycbcr_input = ycbcr_channel(input, self.patch_size)
        ycbcr_target = ycbcr_channel(target, self.patch_size)

        # Compute the loss
        y_loss = F.mse_loss(ycbcr_input[0], ycbcr_target[0])
        cb_loss = F.mse_loss(ycbcr_input[1], ycbcr_target[1])
        cr_loss = F.mse_loss(ycbcr_input[2], ycbcr_target[2])

        # Total loss
        loss = y_loss + cb_loss + cr_loss
        return loss

"""CIELUV Channel Loss"""
class CIELUVChannelLoss(nn.Module):
    def __init__(self,id:int = None, patch_size=15):
        super(CIELUVChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id = id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def rgb_to_cieluv(self, rgb):
        """
        Convert RGB to CIELUV color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tensor: The CIELUV image with shape (N, 3, H, W).
        """
        # Convert RGB to XYZ
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]

        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # Normalize XYZ
        x_n =  0.95047
        y_n =  1.00000
        z_n =  1.08883

        x = x / x_n
        z = z / z_n

        def f(t):
            return torch.where(t > 0.008856, t.pow(1/3), 7.787 * t + 16/116)

        l = 116 * f(y) - 16
        u = 13 * l * (f(x) - f(y))
        v = 13 * l * (f(y) - f(z))

        return torch.stack([l, u, v], dim=1)

    def forward(self, input, target):
        """
        Compute the CIELUV Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The CIELUV Channel Loss value.
        """
        def cieluv_channel(image, patch_size):
            # Compute the CIELUV channels of an image
            cieluv = self.rgb_to_cieluv(image)
            l = cieluv[:, 0, :, :]
            u = cieluv[:, 1, :, :]
            v = cieluv[:, 2, :, :]

            # Apply convolution to compute channel loss (CIELUV-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            l_channel = F.conv2d(l.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            u_channel = F.conv2d(u.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            v_channel = F.conv2d(v.unsqueeze(1), kernel, stride=1, padding=patch_size//2)

            return l_channel, u_channel, v_channel

        # Compute CIELUV channels
        cieluv_input = cieluv_channel(input, self.patch_size)
        cieluv_target = cieluv_channel(target, self.patch_size)

        # Compute the loss
        l_loss = F.mse_loss(cieluv_input[0], cieluv_target[0])
        u_loss = F.mse_loss(cieluv_input[1], cieluv_target[1])
        v_loss = F.mse_loss(cieluv_input[2], cieluv_target[2])

        # Total loss
        loss = l_loss + u_loss + v_loss
        return loss

"""YUV 420 Channel Loss"""
class YUV420ChannelLoss(nn.Module):
    def __init__(self,id=None, patch_size:int = 15):
        super(YUV420ChannelLoss, self).__init__()
        self.patch_size = patch_size
        self._id=id
    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id
    def rgb_to_yuv420(self, rgb):
        """
        Convert RGB to YUV420 color space.
        
        Args:
            rgb (Tensor): The RGB image with shape (N, C, H, W).
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: The Y, U, and V images with shapes (N, 1, H, W), (N, 1, H/2, W/2), (N, 1, H/2, W/2).
        """
        r = rgb[:, 0, :, :]
        g = rgb[:, 1, :, :]
        b = rgb[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.169 * r - 0.331 * g + 0.500 * b + 128
        v = 0.500 * r - 0.460 * g - 0.040 * b + 128

        # Downsample U and V
        u = F.avg_pool2d(u.unsqueeze(1), kernel_size=2, stride=2, padding=0).squeeze(1)
        v = F.avg_pool2d(v.unsqueeze(1), kernel_size=2, stride=2, padding=0).squeeze(1)
        
        return y, u, v

    def forward(self, input, target):
        """
        Compute the YUV420 Channel Loss between the input and target images.
        
        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).
        
        Returns:
            Tensor: The YUV420 Channel Loss value.
        """
        def yuv420_channel(image, patch_size):
            # Compute the YUV420 channels of an image
            y, u, v = self.rgb_to_yuv420(image)
            y_target, u_target, v_target = self.rgb_to_yuv420(target)

            # Apply convolution to compute channel loss (YUV420-based)
            kernel = torch.ones((1, 1, patch_size, patch_size), device=image.device)
            y_channel = F.conv2d(y.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            u_channel = F.conv2d(u.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            v_channel = F.conv2d(v.unsqueeze(1), kernel, stride=1, padding=patch_size//2)

            y_target_channel = F.conv2d(y_target.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            u_target_channel = F.conv2d(u_target.unsqueeze(1), kernel, stride=1, padding=patch_size//2)
            v_target_channel = F.conv2d(v_target.unsqueeze(1), kernel, stride=1, padding=patch_size//2)

            return y_channel, u_channel, v_channel, y_target_channel, u_target_channel, v_target_channel

        # Compute YUV420 channels
        yuv420_input = yuv420_channel(input, self.patch_size)
        yuv420_target = yuv420_channel(target, self.patch_size)

        # Compute the loss
        y_loss = F.mse_loss(yuv420_input[0], yuv420_target[0],reduction='mean')
        u_loss = F.mse_loss(yuv420_input[1], yuv420_target[1],reduction='mean')
        v_loss = F.mse_loss(yuv420_input[2], yuv420_target[2],reduction='mean')

        # Total loss
        loss = y_loss + u_loss + v_loss
        return loss
    
"""Histogram Color Loss"""#%
class HistogramColorLoss(nn.Module):
    def __init__(self, bins:int =256, id: int = None):
        super(HistogramColorLoss, self).__init__()
        self.bins = bins
        self._id=id

    @property
    def name(self):
        return self.__class__.__name__
    @property
    def id(self):
        return self._id

    def compute_histogram(self, image, bins):
        """
        Compute the histogram of an image.

        Args:
            image (Tensor): The image with shape (N, C, H, W).
            bins (int): Number of bins for the histogram.

        Returns:
            Tensor: The histogram with shape (N, C, bins).
        """
        N, C, H, W = image.shape
        histograms = []
        for c in range(C):
            channel = image[:, c, :, :].view(N, -1)  # Flatten the channel
            histogram = torch.histc(channel, bins=bins, min=0, max=1)  # Compute histogram
            histograms.append(histogram)
        
        return torch.stack(histograms, dim=1)  # Shape (N, C, bins)

    @normalize_loss_output
    def forward(self, input, target):
        """
        Compute the histogram color loss between the input and target images.

        Args:
            input (Tensor): The predicted image with shape (N, C, H, W).
            target (Tensor): The ground truth image with shape (N, C, H, W).

        Returns:
            Tensor: The Histogram Color Loss value.
        """
        # Normalize images to [0, 1] range
        input = (input - input.min()) / (input.max() - input.min())
        target = (target - target.min()) / (target.max() - target.min())
        
        # Compute histograms
        hist_input = self.compute_histogram(input, self.bins)
        hist_target = self.compute_histogram(target, self.bins)
        
        # Calculate the histogram loss
        loss = F.mse_loss(hist_input, hist_target,reduction='mean')
        return loss


