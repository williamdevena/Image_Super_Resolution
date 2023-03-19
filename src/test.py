
import os

from torchvision import transforms
from tqdm import tqdm


def testing_generator(model, test_dataloader, device, logger, image_folder, name_for_tensorboard):
    model.to(device)
    len_dataloader = len(test_dataloader)
    pbar = tqdm(test_dataloader)
    tot_lpips = 0
    tot_ssim = 0
    tot_psnr = 0
    pbar.set_description("Testing")
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images = hr_images.squeeze(0)
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)
        fake_hr_images = model(lr_images)
        fake_hr_images = fake_hr_images.squeeze(0)
        fake_hr_images = fake_hr_images.to(device)

        fake_hr_images = (fake_hr_images+1)/2

        #print(hr_images.shape, fake_hr_images.shape)

        #print(fake_hr_images)

        pil_fake_image = transforms.ToPILImage()(fake_hr_images)
        image_path = os.path.join(image_folder, f"{idx}.png")
        pil_fake_image.save(image_path)

        lpips, ssim, psnr = calculate_metrics(hr_images, fake_hr_images, device)
        tot_lpips += lpips
        tot_ssim += ssim
        tot_psnr += psnr

        logger.add_scalars(name_for_tensorboard, {
                'lpips': lpips,
                'ssim': ssim,
                'psnr': psnr,
            }, idx)

        #pbar.set_postfix(L1=loss.item())

    tot_lpips /= len(test_dataloader)
    tot_ssim /= len(test_dataloader)
    tot_psnr /= len(test_dataloader)


    return tot_lpips, tot_ssim, tot_psnr
