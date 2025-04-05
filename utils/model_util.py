import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, Union
import importlib
import os

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE, hf_cache_home
from packaging import version
import sys

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

# from .import_utils import is_peft_available, is_transformers_available
_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False


_peft_available = importlib.util.find_spec("peft") is not None
try:
    _peft_version = importlib_metadata.version("peft")
    logger.debug(f"Successfully imported peft version {_peft_version}")
except importlib_metadata.PackageNotFoundError:
    _peft_available = False

def is_transformers_available():
    return _transformers_available

def is_peft_available():
    return _peft_available



default_cache_path = HUGGINGFACE_HUB_CACHE

MIN_PEFT_VERSION = "0.5.0"
MIN_TRANSFORMERS_VERSION = "4.33.3"


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DIFFUSERS_CACHE = default_cache_path
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]

# from ..utils import USE_PEFT_BACKEND
# from .lora import LoRACompatibleLinear

# Below should be `True` if the current version of `peft` and `transformers` are compatible with
# PEFT backend. Will automatically fall back to PEFT backend if the correct versions of the libraries are
# available.
# For PEFT it is has to be greater than 0.6.0 and for transformers it has to be greater than 4.33.1.
_required_peft_version = is_peft_available() and version.parse(
    version.parse(importlib.metadata.version("peft")).base_version
) > version.parse(MIN_PEFT_VERSION)
_required_transformers_version = is_transformers_available() and version.parse(
    version.parse(importlib.metadata.version("transformers")).base_version
) > version.parse(MIN_TRANSFORMERS_VERSION)

USE_PEFT_BACKEND = _required_peft_version and _required_transformers_version


class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.dtype, fused_weight.device

        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        self.proj = linear_cls(dim_in, dim_out * 2)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, scale: float = 1.0):
        args = () if USE_PEFT_BACKEND else (scale,)
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


ckpt_set = {
    "nudity": {
        "UCE": "nudity/UCE/UCE-Nudity-Diffusers-UNet.pt",
        "ESD": "nudity/ESD/ESD-Nudity-Diffusers-UNet-noxattn.pt",
        "SPM": "nudity/SPM/SPM-Nudity-Diffusers-UNet.pt",
        "FMN": "nudity/FMN/FMN-Nudity-Diffusers-UNet.pt",
        "AdvUnlearn": "nudity/AdvUnlearn/AdvUnlearn_Nudity_text_encoder_full.pt",
        "RECE": "nudity/RECE/nudity_ep2.pt",
        "MACE":"nudity/MACE",
        "ConcptPrune": "nudity/ConcptPrune/Nudity_skilled_neurons_0.01.pt",
        "DoCoPreG" :"nudity/DoCoPreG/Nudity.bin",
        "Receler": "nudity/Receler"
    },
    "style_vangogh": {
        "UCE": "vangogh/UCE/UCE-VanGogh-Diffusers-UNet.pt",
        "ESD": "vangogh/ESD/ESD-VanGogh-Diffusers-UNet-xattn.pt",
        "SPM": "vangogh/SPM/SPM-VanGogh-Diffusers-UNet.pt",
        "FMN": "vangogh/FMN/FMN-VanGogh-Diffusers-UNet.pt",
        "AdvUnlearn": "vangogh/AdvUnlearn/AdvUnlearn_VanGogh_text_encoder_layer0.pt",
        "RECE": "vangogh/RECE/VanGogh_ep0.pt",
        "MACE":"vangogh/MACE",
        "ConcptPrune": "vangogh/ConcptPrune/VanGogh_skilled_neurons_0.01.pt",
        "DoCoPreG" :"vangogh/DoCoPreG/Vangogh.bin",
        "Receler": "vangogh/Receler"
    },
    "object_church": {
        "AdvUnlearn": "object/AdvUnlearn/AdvUnlearn_Church_text_encoder_layer0.pt",  
        "ESD": "object/ESD/ESD-Church-Diffusers-UNet-noxattn.pt",
        "FMN": "object/FMN/FMN-Church-Diffusers-UNet.pt",
        "SPM": "object/SPM/SPM-Church-Diffusers-UNet.pt",
        "RECE": "object/RECE/Church_ep0.pt",
        "MACE":"object/MACE/object_church",
        "UCE": "object/UCE/Church-sd_1_4.pt",
        "ConcptPrune": "object/ConcptPrune/Church_skilled_neurons_0.01.pt",
        "DoCoPreG" :"object/DoCoPreG/Church.bin",
        "Receler": "object/Receler/Church"
    },
    "object_parachute": {
        "AdvUnlearn": "object/AdvUnlearn/AdvUnlearn_Parachute_text_encoder_layer0.pt",  
        "ESD": "object/ESD/ESD-Parachute-Diffusers-UNet-noxattn.pt",
        "FMN": "object/FMN/FMN-Parachute-Diffusers-UNet.pt",
        "SPM": "object/SPM/SPM-Parachute-Diffusers-UNet.pt",
        "RECE": "object/RECE/Parachute_ep0.pt",
        "MACE":"object/MACE/object_parachute",
        "UCE": "object/UCE/Parachute-sd_1_4.pt",
        "ConcptPrune": "object/ConcptPrune/Parachute_skilled_neurons_0.01.pt",
        "DoCoPreG" :"object/DoCoPreG/Parachute.bin",
        "Receler": "object/Receler/Parachute"
    },
}
