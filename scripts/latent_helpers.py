from scripts.helpers import *


def get_conditionings(
    model,
    value_dict,
    num_samples
):
    force_uc_zero_embeddings = []
    additional_batch_uc_fields = []

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                    T=None,
                    additional_batch_uc_fields=additional_batch_uc_fields,
                )

                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                    force_cond_zero_embeddings=None,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )
    return c, uc


def get_samples(
    model,
    sampler,
    num_samples,
    c,
    uc,
    H=1024,
    W=576,
    C=4,
    F=8,
    seed=42
):
    reset_rng(seed)
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                additional_model_inputs = {}
                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)
                samples = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)
    return samples


def decode_samples(
    model,
    samples
): 
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                load_model(model.first_stage_model)
                model.en_and_decode_n_samples_a_time = (
                    None  # Decode n frames at a time
                )
                samples_x = model.decode_first_stage(samples)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)
                grid = torch.stack([samples])
                grid = rearrange(grid, "n b c h w -> (n h) (b w) c")
                return samples