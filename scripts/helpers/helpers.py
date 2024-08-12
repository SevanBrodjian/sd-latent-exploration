from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from sgm.util import instantiate_from_config, default

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch import autocast
from torchvision.utils import make_grid
from sgm.modules.diffusionmodules.discretizer import Discretization

from PIL import Image
import math
import copy
from einops import repeat, rearrange
import os
import numpy as np
import time
from joblib import dump, load
import warnings
import random
import cv2
import gc


def read_prompts(filepath):
    with open(filepath, 'r') as file:
        content = file.read().strip() 
        prompts = content.split('\n\n')
    return prompts


def clear_vram(model = None):
    if model != None:
        del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def reset_rng(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Img2ImgDiscretizationWrapper:
    """
    wraps a discretizer, and prunes the sigmas
    params:
        strength: float between 0.0 and 1.0. 1.0 means full sampling (all sigmas are returned)
    """

    def __init__(self, discretization: Discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        # sigmas start large first, and decrease then
        sigmas = self.discretization(*args, **kwargs)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        sigmas = torch.flip(sigmas, (0,))
        return sigmas


def resize_to_div32(image_tensor):
    _, _, H, W = image_tensor.shape

    new_H = (H // 32) * 32
    if H % 32 != 0:
        new_H += 32

    new_W = (W // 32) * 32
    if W % 32 != 0:
        new_W += 32

    resized_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return resized_tensor


def interpolate_samples(samples, interp):
    N, C, H, W = samples.shape
    assert N > 1, "There should be at least two samples to interpolate between."
    
    new_samples = [samples[0]]
    
    for i in range(N - 1):
        start = samples[i]
        end = samples[i + 1]
        
        for j in range(1, interp + 1):
            alpha = j / (interp + 1)
            interpolated_sample = (1 - alpha) * start + alpha * end
            new_samples.append(interpolated_sample)
    
        new_samples.append(end)
    
    return torch.stack(new_samples)


def interpolate_conds(conds, ucs, interp):
    N = conds['crossattn'].shape[0]
    assert N > 1, "There should be at least two samples to interpolate between."

    interp_conds = {}
    interp_ucs = {}
    
    for condtype in ['vector', 'crossattn']:
        new_conds = [conds[condtype][0]]
        
        for i in range(N - 1):
            start = conds[condtype][i]
            end = conds[condtype][i + 1]
            
            for j in range(1, interp + 1):
                alpha = j / (interp + 1)
                interpolated_cond = (1 - alpha) * start + alpha * end
                new_conds.append(interpolated_cond)
        
            new_conds.append(end)
        
        interp_conds[condtype] = torch.stack(new_conds)
        interp_ucs[condtype] = torch.stack([ucs[condtype][0]]*len(new_conds))
    
    return interp_conds, interp_ucs


def save_png(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        Image.fromarray(sample.astype(np.uint8)).save(
            os.path.join(save_path, f"{base_count:09}.png")
        )
        base_count += 1


def save_mp4(save_path, samples, fps=12):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    video_filename = os.path.join(save_path, f"{base_count:09}.mp4")
    
    height, width = samples.shape[2], samples.shape[3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        frame = sample.astype(np.uint8)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    video_writer.release()


def print_dur(time_start, time_end, message="Time spent"):
    elapsed_time = time_end - time_start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    formatted_message = f"{message}: {minutes}M:{seconds:.1f}S"
    print(formatted_message)


def load_model_from_config(config, ckpt=None, verbose=True, print_time=True):
    start_loadmodel = time.time()
    model = instantiate_from_config(config.model)
    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        if ckpt.endswith("safetensors"):
            sd = load_safetensors(ckpt)
        else:
            raise NotImplementedError

        m, u = model.load_state_dict(sd, strict=False)

        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    model.cuda()
    model.eval()
    end_loadmodel = time.time()
    print_dur(start_loadmodel, end_loadmodel, "Model load time")
    return model


def load_model_from_joblib(path):
    model = load(path)
    model.cuda()
    model.eval()
    warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
    return model


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_image(image_path = None):
    if image_path is not None:
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return image


def load_img(
    image_path = None,
    size = None,
    center_crop = False,
):
    image = get_image(image_path)
    if image is None:
        return None
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")

    transform = []
    if size is not None:
        transform.append(transforms.Resize(size))
    if center_crop:
        transform.append(transforms.CenterCrop(size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Lambda(lambda x: 2.0 * x - 1.0))

    transform = transforms.Compose(transform)
    img = transform(image)[None, ...]
    return img


def init_embedder_options(keys, init_dict, prompt=None, negative_prompt=None, options = None):
    options = {} if options is None else options

    value_dict = {}
    for key in keys:
        if key == "txt":
            if prompt is None:
                prompt = "A professional photograph of an astronaut riding a pig"
            if negative_prompt is None:
                negative_prompt = ""
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt

        if key == "original_size_as_tuple":
            value_dict["orig_width"] = init_dict["orig_width"]
            value_dict["orig_height"] = init_dict["orig_height"]

        if key == "crop_coords_top_left":
            value_dict["crop_coords_top"] = options.get("crop_coords_top", 0)
            value_dict["crop_coords_left"] = options.get("crop_coords_left", 0)

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = options.get("aesthetic_score", 6.0)
            value_dict["negative_aesthetic_score"] = options.get("negative_aesthetic_score", 2.5)

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

        if key in ["fps_id", "fps"]:
            fps = options.get("fps", 6)

            value_dict["fps"] = fps
            value_dict["fps_id"] = fps - 1

        if key == "motion_bucket_id":
            mb_id = options.get("mb_id", 127)
            value_dict["motion_bucket_id"] = mb_id

        if key == "pool_image":
            image = load_img(
                image_path = options.get("image_path", None),
                size=224,
                center_crop=True,
            )
            if image is None:
                image = torch.zeros(1, 3, 224, 224)
            value_dict["pool_image"] = image

    return value_dict


def get_turbo_batch(prompts, dims):
    num_prompts = len(prompts)
    dims_tensor = torch.tensor([dims], device='cuda').repeat(num_prompts, 1)
    batch = {
        'original_size_as_tuple': dims_tensor,
        'txt': prompts,
        'crop_coords_top_left': torch.tensor([[0, 0]], device='cuda').repeat(num_prompts, 1),
        'target_size_as_tuple': dims_tensor
    }
    return batch


def get_batch(
    keys,
    value_dict,
    N,
    device = "cuda",
    T = None,
    additional_batch_uc_fields = [],
):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = [value_dict["prompt"]] * math.prod(N)

            batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(math.prod(N), 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        elif key == "polars_rad":
            batch[key] = torch.tensor(value_dict["polars_rad"]).to(device).repeat(N[0])
        elif key == "azimuths_rad":
            batch[key] = (
                torch.tensor(value_dict["azimuths_rad"]).to(device).repeat(N[0])
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key in additional_batch_uc_fields and key not in batch_uc:
            batch_uc[key] = copy.copy(batch[key])
    return batch, batch_uc


def change_framerate(input_path, output_path, new_fps):
    cap = cv2.VideoCapture(input_path)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()