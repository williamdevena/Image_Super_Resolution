
import lpips
import skimage
import torch
from torchvision import transforms


def calculate_metrics(original_img, upscaled_img):
    """
    Given an original high resolution image and a upscaled
    one, it calculates several metrics to assess the perfomance
    of a upscaling model

    Args:
        - original_img (np.ndarray): original high resolution image
        - upscaled_img (np.ndarray): upscaled image

    Returns:
        - lpips (float): LPIPS (Learned Perceptual Image Patch Similarity) metric
        - ssim (float): SSIM (structural similarity index measure) metric
        - psnr (float): PSNR (peak signal-to-noise ratio) metric
    """
    lpips = calculate_lpips_distance(original_img, upscaled_img)
    ssim = skimage.metrics.structural_similarity(original_img, upscaled_img, channel_axis=2)
    psnr = skimage.metrics.peak_signal_noise_ratio(original_img, upscaled_img)

    return lpips, ssim, psnr


def calculate_lpips_distance(img1, img2):
    """
    Calculates LPIPS distance (also called perceptual loss).
    Reference: "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
    Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. In CVPR, 2018.

    Args:
        - img1 (np.ndarray): fisrt image to compare
        - img2 (np.ndarray): second image to compare

    Returns:
        - lpips_distance (float): LPIPS distance
    """
    img_shape = img1.shape
    img1 = torch.tensor(img1).view(3, img_shape[0], img_shape[1]).float()
    img2 = torch.tensor(img2).view(3, img_shape[0], img_shape[1]).float()

    # mean1, std1 = torch.Tensor.float(img1).mean([1,2]), torch.Tensor.float(img1).std([1,2])
    # mean2, std2 = torch.Tensor.float(img2).mean([1,2]), torch.Tensor.float(img2).std([1,2])
    mean1, std1 = img1.mean([1,2]), img1.std([1,2])
    mean2, std2 = img2.mean([1,2]), img2.std([1,2])

    # define custom transform
    # here we are using our calculated
    # mean & std
    transform1 = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean1, std1)
        #transforms.Normalize()
    ])

    transform2 = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize(mean2, std2)
        #transforms.Normalize()
    ])

    img1 = transform1(img1)
    img2 = transform2(img2)

    #print(img1, img2)

    loss_fn = lpips.LPIPS(net='alex') # best forward scores
    #loss_fn = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    lpips_distance = loss_fn(img1, img2).item()

    return lpips_distance
