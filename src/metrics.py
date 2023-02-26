
import lpips


def calculate_lpips_distance(img1, img2):
    """
    Calculates LPIPS distance (also called perceptual loss).
    Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
    Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. In CVPR, 2018.

    Args:
        img1 (_type_): _description_
        img2 (_type_): _description_

    Returns:
        _type_: _description_
    """

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    #loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    d = loss_fn_alex(img1, img2).item()

    return d
