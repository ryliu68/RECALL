import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
import os
from tqdm.auto import tqdm
from copy import deepcopy

import torchvision.transforms as T

import json
from typing import List, Optional, Union, Tuple


from utils import CustomTextEncoder, ckpt_set, GEGLU, NeuronRemover, inject_eraser

to_pil = T.ToPILImage()
totensor = T.ToTensor()
topil = T.ToPILImage()


ckpt_BASE = F"ckpts/unlearned_ckpt"


class SDAModel(object):
    def __init__(self, unlearn_method, concept,  device, num_inference_steps=50, criterion=None, data_type=torch.float16):
        self.unlearn_method = unlearn_method
        self.concept = concept
        self.device = device
        self.data_type = data_type
        self.criterion = torch.nn.L1Loss() if criterion == 'l1' else torch.nn.MSELoss()

        self.T = 1000
        self.n_samples = num_inference_steps
        start = self.T // self.n_samples // 2
        self.sampled_t = list(range(start, self.T, self.T // self.n_samples))[:self.n_samples]

        self.generator = torch.Generator(device=self.device)

    def set_(self, seed, guidance_scale):
        self.generator.manual_seed(seed)  # Set the seed for reproducibility

        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = self.guidance_scale > 1.0

    def load_DM(self,):
        model_id = "CompVis/stable-diffusion-v1-4"
        #
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,  # put your model path here
            # revision="fp16",
            torch_dtype=self.data_type,
        ).to(self.device)

        #
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.feature_extractor = pipe.feature_extractor
        self.unet_sd = pipe.unet

        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # ############################## GPU ###############################

        if self.unlearn_method == "ORG":
            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method in ["ESD", "FMN", "SPM", "RECE", "UCE"]:
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"
            self.unet = deepcopy(self.unet_sd)
            self.unet.load_state_dict(torch.load(target_ckpt, map_location=self.device))

        elif self.unlearn_method == "AdvUnlearn":
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"

            def extract_text_encoder_ckpt(ckpt_path):
                full_ckpt = torch.load(ckpt_path)
                new_ckpt = {}
                for key in full_ckpt.keys():
                    if 'text_encoder.text_model' in key:
                        new_ckpt[key.replace("text_encoder.", "")] = full_ckpt[key]
                return new_ckpt

            # text_encoder
            self.text_encoder.load_state_dict(extract_text_encoder_ckpt(target_ckpt), strict=False)
            self.text_encoder = self.text_encoder.to(self.device)
            self.custom_text_encoder = CustomTextEncoder(self.text_encoder).to(self.device)
            self.all_embeddings = self.custom_text_encoder.get_all_embedding().unsqueeze(0)

            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "MACE":
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"

            pipe = StableDiffusionPipeline.from_pretrained(target_ckpt, torch_dtype=torch.float16).to(self.device)

            #
            self.vae = pipe.vae
            self.tokenizer = pipe.tokenizer
            self.text_encoder = pipe.text_encoder
            self.unet = pipe.unet

        elif self.unlearn_method == "DoCoPreG":
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"

            #
            st = torch.load(target_ckpt)
            for name, params in self.unet_sd.named_parameters():
                if name in st['unet']:
                    params.data.copy_(st['unet'][f'{name}'])

            self.unet = deepcopy(self.unet_sd)

        elif self.unlearn_method == "ConcptPrune":
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"

            # NeuronRemover
            neuron_remover = NeuronRemover(path_expert_indx=target_ckpt, T=50, n_layers=16, replace_fn=GEGLU, hook_module='unet')

            pipe = neuron_remover.observe_activation(pipe)

            self.unet = deepcopy(pipe.unet)

        elif self.unlearn_method == "Receler":
            target_ckpt = F"{ckpt_BASE}/{ckpt_set[self.concept][self.unlearn_method]}"

            eraser_ckpt_path = os.path.join(target_ckpt, f'eraser_weights.pt')
            eraser_config_path = os.path.join(target_ckpt, f'eraser_config.json')
            with open(eraser_config_path) as f:
                eraser_config = json.load(f)
            # inject erasers into pretrained SD
            inject_eraser(pipe.unet, torch.load(eraser_ckpt_path, map_location='cpu'), **eraser_config)
            pipe.to(self.device, torch_dtype=torch.float16)

            self.unet = deepcopy(pipe.unet)

        else:
            raise ValueError(f"Unlearn method {self.unlearn_method} not supported.")

    def latent2_img(self, latents):
        latents = latents.to(self.device) / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0)

        return to_pil(image)

    def randn_tensor(self,
                     shape: Union[Tuple, List],
                     generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
                     device: Optional["torch.device"] = None,
                     dtype: Optional["torch.dtype"] = None,
                     layout: Optional["torch.layout"] = None,):

        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents

    def prepare_latents(self, image, dtype, device, generator=None):

        image = image.to(device=device, dtype=dtype)

        org_img_init_latents = self.vae.encode(image).latent_dist.sample(generator)

        org_img_init_latents = self.vae.config.scaling_factor * org_img_init_latents

        org_img_init_latents = torch.cat([org_img_init_latents], dim=0)

        return org_img_init_latents

    def _encode_prompt(self,
                       prompt,
                       num_images_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt=None,
                       prompt_embeds: Optional[torch.FloatTensor] = None,
                       negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                       ):

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(self.device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(self.device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def get_img_prompt_latent(self, image, prompt, seed=2025):

        num_images_per_prompt = 1

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance)

        # 4. Preprocess image
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
        image = image_processor.preprocess(image)

        # 6. Prepare latent variables
        org_img_latents = self.prepare_latents(
            image,  prompt_embeds.dtype, self.device, self.generator)

        return org_img_latents, prompt_embeds

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    def gen_image_ti2i_infer(self,
                             prompt=None,
                             image=None,
                             noise=None,
                             strength=0.8,
                             num_images_per_prompt=1,
                             num_inference_steps=50,
                             init_latents=None,
                             prompt_embeds=None,
                             seed=2025):

        if prompt_embeds is None:
            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError
            # 3. Encode input prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        else:
            batch_size = 1

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        if init_latents is None:
            # 4. Preprocess image
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
            image = image_processor.preprocess(image)

            # 6. Prepare latent variables
            org_img_latents = self.prepare_latents(image, prompt_embeds.dtype, self.device)

            latents = org_img_latents.clone()

        else:
            latents = init_latents.repeat(batch_size, 1, 1, 1)

        if noise is None:
            noise = self.randn_tensor(latents.shape, generator=self.generator, device=self.device, dtype=prompt_embeds.dtype)
        else:
            noise = noise

        # latents
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents = self.scheduler.add_noise(latents, noise, latent_timestep)

        # 8. Denoising loop
        for i, t in tqdm(enumerate(timesteps), disable=True):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False)[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents,  return_dict=False)[0]

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        return image, noise
