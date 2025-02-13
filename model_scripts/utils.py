import re
import os

import torch
import numpy as np

import cv2

from diffusers import AsymmetricAutoencoderKL, StableDiffusionInpaintPipeline, StableDiffusionXLPipeline, \
    StableDiffusionXLControlNetPipeline,  ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler, ConsistencyDecoderVAE
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import *

from model_scripts.inpaint_fill_cn import StableDiffusionControlNetInpaintPipelinev2
from model_scripts.sd_repaint_pipeline_soft import StableDiffusionRepaintPipelineSoft
from model_scripts.repaint_scheduler_soft import RePaintSchedulerSoft
from model_scripts.sd_inpaint_pipeline_soft import StableDiffusionInpaintPipelineSoft
from model_scripts.diff_diff_sd2 import StableDiffusionDiffImg2ImgPipeline

T_THRESH = float(os.getenv("T_THRESH", 0.05))
STRENGTH = float(os.getenv("STRENGTH", 1.0))
PROMPT = os.getenv("PROMPT", "")
vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

def get_sd_pipeline(device, strategy):
    pipes = []
    if re.match(".*cnSDXL\+diffdiff", strategy) is not None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img"
        )
        pipe.enable_model_cpu_offload()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, solver_order=1)
        pipes.append(pipe)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.insert(0, pipe)

    elif re.match(".*cnSDXLinp\+diffdiff", strategy) is not None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img"
        )
        pipe.enable_model_cpu_offload()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, solver_order=1)
        pipes.append(pipe)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.insert(0, pipe)

    if re.match(".*cnSD\+diffdiff", strategy) is not None:
        pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16", vae=vae
        )
        pipe.enable_model_cpu_offload()
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, solver_order=1)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)
        pipes.append(pipe)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
            vae=vae
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.insert(0, pipe)

    elif re.match(".*cnSDinp\+diffdiff", strategy) is not None:
        pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, solver_order=1)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=False)
        pipes.append(pipe)

        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.insert(0, pipe)

    elif re.match(".*cnSD", strategy):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.append(pipe)
    
    elif re.match(".*cnSDinp", strategy):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16"
        )
        pipe = StableDiffusionControlNetInpaintPipelinev2.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            variant="fp16",
            controlnet=controlnet,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipes.append(pipe)

    elif (strategy == "pm+diffdiff") or (strategy == "telea+diffdiff") or (strategy == "diffdiff"):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img"
        )
        pipe.enable_model_cpu_offload()
        pipes.append(pipe)

    elif strategy == "SDInp":
        pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        # pipe.enable_model_cpu_offload()
        pipes.append(pipe)

    elif strategy == "repaint":
        pipe = StableDiffusionRepaintPipelineSoft.from_pretrained("stabilityai/stable-diffusion-2-base",#"stabilityai/stable-diffusion-2",
                                                                torch_dtype=torch.float16).to(device)
        pipe.scheduler = RePaintSchedulerSoft.from_config(pipe.scheduler.config)
        pipe.scheduler.eta = 1.0
        pipes.append(pipe)

    elif (strategy == "pm") or (strategy == "telea"):
        pipe = NonLearnedInpainting(strategy)
        pipes.append(pipe)

    else:
        # This is from invisible stitch https://github.com/paulengstler/invisible-stitch/blob/09cd176ef72fd0a088d432745f831b962690b2a3/utils/models.py#L109
        pipe = StableDiffusionInpaintPipelineSoft.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to(device)
        pipe.vae = AsymmetricAutoencoderKL.from_pretrained(
            "cross-attention/asymmetric-autoencoder-kl-x-2",
            torch_dtype=torch.float16
        ).to(device)
        pipes.append(pipe)

    return pipes

def process_inputs(camera, pipe, im=None, strategy=""):
    if pipe.__class__.__name__ == "StableDiffusionXLDifferentialImg2ImgPipeline" or isinstance(pipe, StableDiffusionDiffImg2ImgPipeline):
        if isinstance(pipe, StableDiffusionDiffImg2ImgPipeline):
            input_size = (512, 512)
        else:
            input_size = (1024, 1024)

        if im is None:
            im = camera.original_image

        # if strategy is of the form [telea,pm]+diffdiff
        strategy_split = strategy.split("+")
        if len(strategy_split) == 2 and im is None:
            mode = strategy_split[0]
            inpaint = NonLearnedInpainting(mode)(camera).images[0]
        else:
            inpaint = im
        mask = camera.original_mask

        inpaint = torch.nn.functional.interpolate(inpaint[None], input_size, mode='bilinear')[0] if inpaint.size(0) !=4 else inpaint
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='bilinear')[0]
        inpaint_ = pipe.image_processor.preprocess(inpaint)
        mask = torch.clamp(mask, 0, 0.8)
        kwargs = {"prompt": "", "image": inpaint_, "original_image": inpaint_,
                  "map": (1-mask), "guidance_scale": 1, "strength": 1.0, "output_type": "pt"}
        if isinstance(pipe, StableDiffusionDiffImg2ImgPipeline):
            kwargs.pop("original_image")
        return inpaint, (1-mask), kwargs

    elif isinstance(pipe, (StableDiffusionRepaintPipelineSoft)):
        input_size = (512, 512)
        im = camera.original_image.to(pipe.device)
        mask = camera.original_mask.to(pipe.device)
        im = torch.nn.functional.interpolate(im[None], input_size, mode='bilinear')[0]
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='bilinear')[0]

        kwargs = {"prompt": "", "image": im, "mask_image": (1-mask)}

        return im, (1-mask), kwargs

    elif isinstance(pipe, StableDiffusionInpaintPipelineSoft):
        input_size = (512, 512)
        im = camera.original_image.to(pipe.device)
        mask = camera.original_mask.to(pipe.device)
        im = torch.nn.functional.interpolate(im[None], input_size, mode='bilinear')[0]
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='bilinear')[0]

        kwargs = {"prompt": "", "image": im, "mask_image": mask, "strength": 1.0, "output_type": "pt"}
        return im, mask, kwargs
    
    elif isinstance(pipe, StableDiffusionInpaintPipeline):
        input_size = (512, 512)
        im = camera.original_image.to(pipe.device)
        mask = camera.original_mask.to(pipe.device)
        im = torch.nn.functional.interpolate(im[None], input_size, mode='bilinear')[0]
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='bilinear')[0]
        mask = torch.where(mask > T_THRESH, 1, 0.)
        im = fill_image(im, mask, "telea")
        # tweaked mask
        grow_mask_ = grow_mask_fn(mask)[None]

        kwargs = {"prompt": "empty", "image": im, "mask_image": grow_mask_, "strength": 0.9, "output_type": "pt"}
        return im, grow_mask_, kwargs

    elif isinstance(pipe, (StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetInpaintPipelinev2)):
        input_size = (512, 512)
        im = camera.original_image
        mask_og = camera.original_mask
        mask = camera.original_mask.cpu().numpy()

        strategy_split = strategy.split("+")
        if strategy_split[0] == "pm":
            fill_mode = "pm"
        elif strategy_split[0] == "telea":
            fill_mode = "telea"
        else:
            fill_mode = "nothing"

        mask = mask > T_THRESH
        mask = torch.from_numpy(mask).to(camera.original_image).type(torch.float32)
        im = torch.nn.functional.interpolate(im[None], input_size, mode='bilinear')[0]
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='nearest-exact')[0]
        im = fill_image(im, mask, fill_mode)

        # tweaked mask
        grow_mask_ = grow_mask_fn(mask)

        control_image = torch.where(grow_mask_ > 0, -1, im)

        kwargs = {"prompt": "", "negative_prompt": "",
                  "controlnet_conditioning_scale": 0.9 if fill_mode != "nothing" else 1.0,
                  "guidance_scale": 0.0, "num_inference_steps": 25, "strength": 0.9 if fill_mode != "nothing" else 1.0, 
                  "image": im[None], "mask_image": grow_mask_[None],
                  "control_image": control_image[None], "output_type": "pt"}
        return im, mask, kwargs

    elif isinstance(pipe, StableDiffusionXLControlNetPipeline):
        input_size = (1024, 1024)
        im = camera.original_image
        mask_og = camera.original_mask
        mask = camera.original_mask.cpu().numpy()
        mask = dilation(mask, np.ones((21, 21)))
        mask = mask > T_THRESH
        mask = torch.from_numpy(mask).to(camera.original_image).type(torch.float32)
        im = torch.nn.functional.interpolate(im[None], input_size, mode='bilinear')[0]
        mask = torch.nn.functional.interpolate(mask[None, None], input_size, mode='nearest-exact')[0]
        mask_og = torch.nn.functional.interpolate(mask_og[None, None], input_size, mode='bilinear')[0]
        control_image = torch.where(mask_og > T_THRESH, 1, im)

        kwargs = {"prompt": "", "negative_prompt": "", "controlnet_conditioning_scale": 0.9,
                  "guidance_scale": 0.0, "num_inference_steps": 25,
                  "image": control_image[None], "control_guidance_end": 0.9}
        return im, mask, kwargs

    elif isinstance(pipe, NonLearnedInpainting):
        im = camera.original_image
        mask = camera.original_mask.cpu().numpy()
        kwargs = {"camera": camera}
        return im, mask, kwargs
    else:
        raise "unknown pipeline"
    
def fill_image(init_image, mask, mode="nothing"):
    if mode == "nothing":
        return init_image
    init_image_np = init_image.squeeze().cpu().permute(1, 2, 0).numpy()

    init_image_np = (init_image_np * 255).astype(np.uint8)
    mask_np = (mask.cpu().squeeze().numpy()*255).astype(np.uint8)

    if mode == "telea":
        masked_image = cv2.inpaint(init_image_np, mask_np, 3, cv2.INPAINT_TELEA)
    else:
        raise "Unknown inpaintinf method"

    masked_image = masked_image / 255.
    masked_image = torch.from_numpy(masked_image).permute(2, 0, 1).expand_as(init_image).to(init_image)

    return masked_image

def grow_mask_fn(mask):
    device = mask.device
    input_shape = mask.shape
    if isinstance(mask, torch.Tensor):
        mask = (mask.cpu().squeeze().numpy()*255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    m2 = mask
    m2 = cv2.erode(m2, kernel, iterations=1)
    m2 = cv2.dilate(m2, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(m2, contours, -1, 255, thickness=cv2.FILLED)

    m2 = m2/255.
    m2 = torch.from_numpy(m2).to(device)
    m2 = m2.reshape(input_shape)
    return m2

class NonLearnedInpainting:
    def __init__(self, inp_mode="telea"):
        self.inp_mode = inp_mode

    def __call__(self, camera=None):
        im = camera.original_image.cpu().permute(1, 2, 0).numpy()
        im = (im * 255).astype(np.uint8)
        mask = camera.inpaint_mask.cpu().numpy()

        mask = dilation(mask, np.ones((21, 21)))
        mask_byte = (mask*255).astype(np.uint8)
        _, mask_np = cv2.threshold(mask_byte, int(T_THRESH*255), 255, cv2.THRESH_BINARY)

        if self.inp_mode == "pm":
            im[mask_np.astype(bool)] = 255
            im = patch_match.inpaint(im, patch_size=3)
        elif self.inp_mode == "telea":
            im = cv2.inpaint(im, mask_np, 3, cv2.INPAINT_TELEA)
        else:
            raise "Unknown inpainting method"

        out = torch.from_numpy(im / 255.).to(camera.original_image).permute(2, 0, 1)
        return NonLearnedInpaintOutput(images=[out], masks=[mask_np])


@dataclass
class NonLearnedInpaintOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    masks: Union[List[PIL.Image.Image], np.ndarray]

def get_modified_nmask(settings, nmask, sigma):
    """
    Converts a negative mask representing the transparency of the original latent vectors being overlaid
    to a mask that is scaled according to the denoising strength for this step.

    Where:
        0 = fully opaque, infinite density, fully masked
        1 = fully transparent, zero density, fully unmasked

    We bring this transparency to a power, as this allows one to simulate N number of blending operations
    where N can be any positive real value. Using this one can control the balance of influence between
    the denoiser and the original latents according to the sigma value.

    NOTE: "mask" is not used
    """
    import torch
    return torch.pow(nmask, (sigma ** settings.mask_blend_power) * settings.mask_blend_scale)


def latent_blend(settings, a, b, t):
    """
    Interpolates two latent image representations according to the parameter t,
    where the interpolated vectors' magnitudes are also interpolated separately.
    The "detail_preservation" factor biases the magnitude interpolation towards
    the larger of the two magnitudes.
    """
    import torch

    # NOTE: We use inplace operations wherever possible.

    if len(t.shape) == 3:
        # [4][w][h] to [1][4][w][h]
        t2 = t.unsqueeze(0)
        # [4][w][h] to [1][1][w][h] - the [4] seem redundant.
        t3 = t[0].unsqueeze(0).unsqueeze(0)
    else:
        t2 = t
        t3 = t[:, 0][:, None]

    one_minus_t2 = 1 - t2
    one_minus_t3 = 1 - t3

    # Linearly interpolate the image vectors.
    a_scaled = a * one_minus_t2
    b_scaled = b * t2
    image_interp = a_scaled
    image_interp.add_(b_scaled)
    result_type = image_interp.dtype
    del a_scaled, b_scaled, t2, one_minus_t2

    # Calculate the magnitude of the interpolated vectors. (We will remove this magnitude.)
    # 64-bit operations are used here to allow large exponents.
    current_magnitude = torch.norm(image_interp, p=2, dim=1, keepdim=True).to(torch.float64).add_(0.00001)

    # Interpolate the powered magnitudes, then un-power them (bring them back to a power of 1).
    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True).to(torch.float64).pow_(
        settings.inpaint_detail_preservation) * one_minus_t3
    b_magnitude = torch.norm(b, p=2, dim=1, keepdim=True).to(torch.float64).pow_(
        settings.inpaint_detail_preservation) * t3
    desired_magnitude = a_magnitude
    desired_magnitude.add_(b_magnitude).pow_(1 / settings.inpaint_detail_preservation)
    del a_magnitude, b_magnitude, t3, one_minus_t3

    # Change the linearly interpolated image vectors' magnitudes to the value we want.
    # This is the last 64-bit operation.
    image_interp_scaling_factor = desired_magnitude
    image_interp_scaling_factor.div_(current_magnitude)
    image_interp_scaling_factor = image_interp_scaling_factor.to(result_type)
    image_interp_scaled = image_interp
    image_interp_scaled.mul_(image_interp_scaling_factor)
    del current_magnitude
    del desired_magnitude
    del image_interp
    del image_interp_scaling_factor
    del result_type

    return image_interp_scaled


class SoftInpaintingSettings:
    def __init__(self,
                 mask_blend_power=1,
                 mask_blend_scale=0.5,
                 inpaint_detail_preservation=4,
                 composite_mask_influence=0,
                 composite_difference_threshold=0.5,
                 composite_difference_contrast=2):
        self.mask_blend_power = mask_blend_power
        self.mask_blend_scale = mask_blend_scale
        self.inpaint_detail_preservation = inpaint_detail_preservation
        self.composite_mask_influence = composite_mask_influence
        self.composite_difference_threshold = composite_difference_threshold
        self.composite_difference_contrast = composite_difference_contrast
