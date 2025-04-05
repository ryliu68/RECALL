import torch
import torch.nn.functional as F
import logging
import os
import argparse
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
import pathlib
from PIL import Image, ImageDraw
from models import SDAModel
from evaluater import Evaluator
from util import set_seed, multidict
import torchvision.transforms as transforms
import json
from data.datasets import PNGImageDataset
import numpy as np
import copy
import shutil
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

to_pil = T.ToPILImage()

DATA_TYPE = torch.float16


def resize_tensor(image_tensor, size=(224, 224)):
    if image_tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), but got shape {image_tensor.shape}")

    return F.interpolate(image_tensor, size=size, mode="bilinear", align_corners=False)


def eval_attack(args, latents, gen_images):
    with torch.no_grad():
        gen_image, _ = args.pipe_attack.gen_image_ti2i_infer(prompt_embeds=args.prompt_embeds,
                                                             image=None,
                                                             noise=None,
                                                             num_inference_steps=args.num_inference_steps,
                                                             init_latents=latents)

        gen_images.append(gen_image)

        # save intermediate generated images
        _gen_images = torch.vstack(gen_images)
        _gen_images = resize_tensor(_gen_images)
        gen_img_save_name = os.path.join(args.saved_path, "gen_imgs", F"{args.attack_id}.png")
        save_image(_gen_images, gen_img_save_name, nrow=10)

    #
    gen_image = to_pil(gen_image.squeeze(0))
    results = args.evaluator.eval(gen_image)

    results["gen_images"] = gen_images
    results["gen_image"] = gen_image

    return results


def loss_adv(self, x0, adv_0, t, encoder_hidden_states, **kwargs):
    noise = self.randn_tensor(x0.shape, generator=self.generator, device=self.device, dtype=x0.dtype)

    # x0
    noised_latent = x0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
        noise * ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)

    noised_latent = noised_latent.to(dtype=x0.dtype)

    noise_pred_x0 = self.unet(torch.cat([noised_latent] * 2), t, encoder_hidden_states=encoder_hidden_states).sample

    # adv_0
    noised_latent = adv_0 * (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(self.device) + \
        noise * ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(self.device)

    noised_latent = noised_latent.to(dtype=adv_0.dtype)

    noise_pred_adv_0 = self.unet(torch.cat([noised_latent] * 2), t, encoder_hidden_states=encoder_hidden_states).sample

    error = self.criterion(noise_pred_x0, noise_pred_adv_0)

    return error


def pgd_attack_with_momentum(args, pipe_attack):
    """
    PGD attack with momentum on latent space.

    Args:
        args: contains prompt_embeds, init_latents, iter, lr, blance, set_each_step
        pipe_attack: provides get_loss_adv_v1 and sampled_t

    Returns:
        adv_latents: list of adversarial latents
        gen_images: list of generated images (optional)
    """

    encoder_hidden_states = args.prompt_embeds
    latents = args.init_latents.clone()
    steps = pipe_attack.sampled_t
    alpha = args.lr
    momentum = 0.9

    gen_images = []
    adv_latents = []

    iterator = tqdm(range(args.iter), desc="Optimizing", disable=True)

    adv_latent = latents*args.blance + torch.randn_like(latents)*(1-args.blance)
    adv_latent = adv_latent.requires_grad_(True)

    velocity = torch.zeros_like(latents)

    for i in range(len(steps)):
        step = steps[i]

        if args.set_each_step:
            adv_latent = latents*args.blance + torch.randn_like(latents)*(1-args.blance)
            adv_latent = adv_latent.requires_grad_(True)
            velocity = torch.zeros_like(latents)

        for iter in iterator:
            loss = loss_adv(pipe_attack, x0=latents, adv_0=adv_latent, t=step, encoder_hidden_states=encoder_hidden_states)

            grads = torch.autograd.grad(loss, adv_latent)[0]
            grads = grads / (grads.abs().mean() + 1e-8)  # normalize

            # Momentum update
            velocity = momentum * velocity + grads
            adv_latent = adv_latent + alpha * velocity.sign()


        if (i+1) % args.interval == 0:
            adv_latent.data += latents*0.05

        adv_latents.append(adv_latent.detach().cpu())

        results = eval_attack(args, adv_latent, gen_images)

        results["adv_latents"] = adv_latents
        results["step"] = i

        if results['success']:

            torch.cuda.empty_cache()
            del gen_images, adv_latents, adv_latent

            return results

    torch.cuda.empty_cache()
    del gen_images, adv_latents,  adv_latent

    return results


def main(args):

    # load prompts
    dataloader = PNGImageDataset(args.concept)

    # ** logging
    saved_path = f"{args.save_path}/{args.concept}/{args.unlearn_method}/" + f"Iter_{args.iter}_lr_{args.lr}_b_{args.blance}_each_{args.set_each_step}/"
    print("saved_path: ", saved_path)
    pathlib.Path(saved_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(F"{saved_path}/adv").mkdir(parents=True, exist_ok=True)
    pathlib.Path(F"{saved_path}/adv_latents").mkdir(parents=True, exist_ok=True)
    pathlib.Path(F"{saved_path}/gen_imgs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(F"{saved_path}/adv_noise").mkdir(parents=True, exist_ok=True)
    pathlib.Path(F"{saved_path}/gen_img").mkdir(parents=True, exist_ok=True)

    shutil.copy("igmu_attack.py", F"{saved_path}/code.py")

    args.saved_path = saved_path

    log_file = os.path.join(saved_path, "attack.log")

    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info(args)

    set_seed(args.seed)

    # load model
    pipe_attack = SDAModel(args.unlearn_method, args.concept, device, data_type=DATA_TYPE)
    pipe_attack.load_DM()

    args.pipe_attack = pipe_attack
    args.evaluator = Evaluator(concept=args.concept, device=device)

    succ_init = 0
    succ = 0
    total_imgs = len(dataloader)
    save_pth = multidict()

    # image
    image_name = F"data/init_image/{args.concept}/x.png"
    init_image = Image.open(image_name).convert('RGB').resize((512, 512))
    args.init_image = init_image

    for idx, data in enumerate(dataloader):
        args.image, args.prompt, seed, args.guidance_scale = data

        if seed is not None:
            args.seed = int(seed)

        pipe_attack.set_(seed=args.seed, guidance_scale=args.guidance_scale)

        # print(f"attack image id: {idx}/{total_imgs}, prompt: {args.prompt}")
        logging.info(f"\nAttack image id: {idx}/{total_imgs}, prompt: {args.prompt}")
        attack_successed = False
        successed_init = False
        args.attack_id = idx
        save_pth[idx]["successed"] = False
        save_pth[idx]["prompt"] = args.prompt
        save_pth[idx]["seed"] = args.seed
        save_pth[idx]["guidance_scale"] = args.guidance_scale

        org_img_latents, prompt_embeds = pipe_attack.get_img_prompt_latent(image=init_image, prompt=args.prompt, seed=args.seed)

        args.init_latents = org_img_latents
        args.prompt_embeds = prompt_embeds

        #
        save_pth[idx]["init_img"] = init_image
        save_pth[idx]["init_latent"] = org_img_latents.detach().cpu()
        save_pth[idx]["promt"] = args.prompt
        save_pth[idx]["prompt_embed"] = args.prompt_embeds.detach().cpu()

        time_start = time.perf_counter()

        #  test init attacted or not START
        with torch.no_grad():
            image_nat, _ = pipe_attack.gen_image_ti2i_infer(
                prompt_embeds=args.prompt_embeds,
                noise=None,
                image=init_image,
                num_inference_steps=50,
                init_latents=None)
            #
            save_pth[idx]["init_gen_img"] = image_nat.detach().cpu()

            image_nat = to_pil(image_nat.squeeze(0))
            results = args.evaluator.eval(image_nat)

        if results['success']:
            succ_init += 1
            succ += 1
            attack_successed = True
            successed_init = True
            adv_image = args.init_image
            save_pth[idx]["successed"] = True
            save_pth[idx]["step"] = 0
            image_adv = image_nat

            logging.info(F"attack success at Init ")
            if args.concept == "nudity":
                print(results['nude'])
                logging.info(results['nude'])
                save_pth[idx]["nude"] = results['nude']

        else:
            results = pgd_attack_with_momentum(args, pipe_attack=pipe_attack)

            save_pth[idx]["step"] = results['step']

            if results['success']:
                attack_successed = True
                succ += 1
                save_pth[idx]["successed"] = True
                if args.concept == "nudity":
                    print(results['nude'])
                    logging.info(results['nude'])

            adv_image = pipe_attack.latent2_img(results["adv_latents"][-1])
            image_adv = results["gen_image"]

            # save intermediate latents
            adv_latents = torch.vstack(results['adv_latents']).detach().cpu()
            adv_latents_save_name = os.path.join(saved_path, "adv_latents", "adv_latents_"+str(idx)+".pth")
            torch.save(adv_latents, adv_latents_save_name)

        #  test init attacted or not END
        time_end = time.perf_counter()

        time_eplase = time_end-time_start
        time_eplase = round(time_eplase, 2)

        if successed_init:
            save_pth[idx]["gen_latent"] = org_img_latents.detach().cpu()
            save_pth[idx]["init_attacked"] = True
        else:
            save_pth[idx]["gen_latent"] = results['adv_latents']
            save_pth[idx]["init_attacked"] = False

        save_pth[idx]["time_eplase"] = time_eplase

        #
        save_pth[idx]["adv_img"] = adv_image
        save_pth[idx]["gen_img"] = image_adv

        logging.info(F"Idx: {idx}\tInit - {succ_init/(idx+1):.4f}\tASR - {succ/(idx+1):.4f}\ttime -  {time_eplase}\tattack_step - {save_pth[idx]['step']}")

        print(F"Idx: {idx}\tInit - {succ_init/(idx+1):.4f}\tASR - {succ/(idx+1):.4f}\ttime -  {time_eplase}\tattack_step - {save_pth[idx]['step']}")

        output_image = Image.new("RGB", (1024, 1024))  # RGB, size=1024x1024

        #
        output_image.paste(init_image, (0, 0))
        output_image.paste(adv_image, (512, 0))
        output_image.paste(image_nat, (0, 512))
        output_image.paste(image_adv, (512, 512))

        #
        draw = ImageDraw.Draw(output_image)

        # add text
        draw.text((10, 10), 'Source Image', fill="white")
        draw.text((522, 10), 'Adv Image', fill="white")
        draw.text((10, 522), 'Gen. Image Nat.', fill="white")
        draw.text((522, 522), 'Gen. Image Adv.', fill="white")

        adv_save_name = os.path.join(saved_path, f"adv/")

        if attack_successed:
            if successed_init:
                output_image.save(adv_save_name+F"{idx}_SUCCESS_init.png")
                image_adv.save(F"{saved_path}/gen_img/{idx}_init.png")
                adv_image.save(F"{saved_path}/adv_noise/{idx}_init.png")
            else:
                output_image.save(adv_save_name+F"{idx}_SUCCESS.png")
                image_adv.save(F"{saved_path}/gen_img/{idx}.png")
                adv_image.save(F"{saved_path}/adv_noise/{idx}.png")

        else:
            output_image.save(adv_save_name+F"{idx}_fail.png")

        torch.save(save_pth, F"{args.saved_path}/results.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for SD attack")
    parser.add_argument("--iter", type=int, default=20, required=False)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--blance", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-grad_acc", "--grad_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument('-s', "--seed", type=int, default=2025, required=False)
    parser.add_argument('-n', "--num_inference_steps", type=int, default=50, required=False)
    parser.add_argument('-un', "--unlearn_method", type=str, default="ESD")
    parser.add_argument('-c', "--concept", type=str, default="object_church", choices=["nudity", "style_vangogh", "object_church", "object_parachute"])
    parser.add_argument('-set-each', "--set_each_step", action="store_true", help="Enable set_each_step")
    args = parser.parse_args()
    print(args)

    main(args)
