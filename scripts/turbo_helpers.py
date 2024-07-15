from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from scripts.helpers import *


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
        ]
        uc = cond
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc
    

def load_fp16():
    model = load("./checkpoints/sdxlturbo_cache.joblib")
    model.eval()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    sampler.n_sample_steps = 1
    return model, sampler


def get_turbo_conditionings(
    model,
    dims,
    prompts
):
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            batch = get_turbo_batch(prompts, dims)
            c = model.conditioner(batch)
    return c


def get_turbo_samples(
    model,
    dims,
    sampler,
    c,
    uc = None
):
    F = 8
    C = 4
    N = c['crossattn'].shape[0]
    shape = (N, C, dims[0] // F, dims[1] // F)
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            randn = torch.randn(shape, device="cuda", dtype=torch.float32)
            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                )
            samples = sampler(denoiser, randn, cond=c, uc=uc)
    return samples


def turbo_decode(
    model,
    samples
):
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            time_start = time.time()
            out = model.decode_first_stage(samples)
            time_end = time.time()
            print_dur(time_start, time_end, "decoding time")
            out_clamped = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)
    return out_clamped


def decode_in_chunks(
    model, 
    samples, 
    chunk_size=6
):
    results = []
    for i in range(0, samples.shape[0], chunk_size):
        chunk = samples[i:i+chunk_size]
        temp = turbo_decode(model, chunk)
        results.append(temp)
        torch.cuda.empty_cache()

    out = torch.cat(results, dim=0)
    return out


def turbo_imgs(
    model,
    sampler,
    prompts,
    dims
):
    F = 8
    C = 4
    N = len(prompts)
    shape = (N, C, dims[0] // F, dims[0] // F)
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            batch = get_turbo_batch(prompts, dims)
            c = model.conditioner(batch)

            randn = torch.randn(shape, device="cuda", dtype=torch.float32)
            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model,
                    input,
                    sigma,
                    c,
                )
            samples = sampler(denoiser, randn, cond=c, uc=None)

            out = model.decode_first_stage(samples)
            out_clamped = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)
    return out_clamped

