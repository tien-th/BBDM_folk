from inspect import isfunction


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def dice_loss(pred, target):
    smooth = 1.0  
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice
    
def get_mask(x, mask_threshold):
    mask = x.clone()
    mask = mask.mul_(0.5).add_(0.5).mul(255.).clamp_(0, 255.)
    mask = (mask > mask_threshold).float()  
    return mask