#!/usr/bin/env python3
"""Record which aten ops real networks run on MPS, with shapes, dtypes and layouts.

The benchmark groups in pmp/ops pick shapes by hand. This tool is the check on
those picks: it runs popular LLMs, encoders, vision, audio and diffusion models
eagerly on MPS under a TorchDispatchMode and logs every op that touches memory,
keyed by (op, input shapes, dtypes, layouts, small non-tensor args). The output
feeds tools/workloads_md.py, which renders WORKLOADS.md.

Weights are random: shapes, dtypes and layouts do not depend on them, and it
keeps the tool offline -- configs come from the local Hugging Face cache or from
hard-coded dimensions, never from a download. Each model runs in the dtype it
ships in (config `torch_dtype`), since that is what users run.

Needs torch, transformers, torchvision and (for diffusion) diffusers, so unlike
the rest of pmp it runs in one fully-equipped env rather than across versions:

    python tools/trace_models.py                 # all models
    python tools/trace_models.py --models llama-3.2-1b resnet50
    python tools/trace_models.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# scipy's dylib does not load on macOS 27 and transformers/diffusers import it
# opportunistically; blocking it makes those imports take their fallback path.
for _mod in ("scipy", "sklearn"):
    sys.modules.setdefault(_mod, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "workloads"
DEV = "mps"

DTYPE_SHORT = {
    torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int64: "i64", torch.int32: "i32", torch.int16: "i16", torch.int8: "i8",
    torch.uint8: "u8", torch.bool: "bool", torch.float64: "f64",
    torch.complex64: "c64",
}

# Ops that do no device work worth benchmarking even though they are not views.
SKIP = {
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided",
    "_to_copy_meta", "lift_fresh", "detach", "alias", "set_", "resize_",
    "_has_compatible_shallow_copy_type", "is_same_size", "sym_size", "sym_stride",
    "sym_numel", "sym_storage_offset", "record_stream",
    # Reshapes whose schema carries no alias annotation but which never copy.
    "_unsafe_view", "_reshape_alias", "as_strided_",
}


def _dense_permutation(t: torch.Tensor) -> bool:
    """True if `t` is some permutation of a contiguous tensor (no gaps, no overlap)."""
    dims = sorted((st, n) for st, n in zip(t.stride(), t.shape) if n != 1)
    expected = 1
    for st, n in dims:
        if st != expected:
            return False
        expected *= n
    return True

# Ops that read a few rows of their first input; see OpRecorder for how they
# are charged.
GATHER_OPS = {"embedding", "index_select", "index", "gather", "take"}


def layout_of(t: torch.Tensor) -> str:
    """Classify the memory layout the kernel sees.

    The order matters: a broadcast operand is reported as such even if it is
    also sliced, because stride-0 is what changes the kernel path first.
    """
    if t.dim() == 0:
        return "scalar"
    if t.is_contiguous():
        return "dense"
    if any(s == 0 and n > 1 for s, n in zip(t.stride(), t.shape)):
        return "bcast"
    if t.dim() == 4 and t.is_contiguous(memory_format=torch.channels_last):
        return "cl"
    if t.dim() == 5 and t.is_contiguous(memory_format=torch.channels_last_3d):
        return "cl3d"
    if _dense_permutation(t):
        # Distinguish the common "last two dims swapped" (x.t(), k.transpose(-1,-2))
        # from general permutations (attention head shuffles).
        if t.dim() >= 2 and t.stride(-2) == 1:
            return "T"
        return "perm"
    return "strided"


def tensor_sig(t: torch.Tensor) -> str:
    dt = DTYPE_SHORT.get(t.dtype, str(t.dtype).removeprefix("torch."))
    lay = layout_of(t)
    shape = ",".join(str(s) for s in t.shape)
    return f"{dt}[{shape}]" + ("" if lay in ("dense", "scalar") else f"{{{lay}}}")


def arg_sig(a) -> str | None:
    """Small non-tensor args (dims, strides, flags) change the kernel; keep them."""
    if isinstance(a, bool) or a is None:
        return None if a is None else str(a)
    if isinstance(a, (int, float)):
        return repr(a)
    if isinstance(a, (list, tuple)) and len(a) <= 8 and all(
            isinstance(x, (int, float, bool)) for x in a):
        return "[" + ",".join(repr(x) for x in a) + "]"
    if isinstance(a, torch.dtype):
        return DTYPE_SHORT.get(a, str(a))
    if isinstance(a, torch.memory_format):
        return str(a).removeprefix("torch.")
    return None


def is_view(func) -> bool:
    rets = func._schema.returns
    return bool(rets) and all(
        r.alias_info is not None and not r.alias_info.is_write for r in rets)


class OpRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.sigs: dict[tuple, dict] = {}
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if name in SKIP or is_view(func) or func.namespace == "profiler":
            return out
        ins = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
        if ins and all(x.device.type != DEV for x in ins):
            return out  # host-side bookkeeping (e.g. position ids built on CPU)
        outs = [x for x in tree_flatten(out)[0] if isinstance(x, torch.Tensor)]
        extras = [s for s in (arg_sig(a) for a in args if not isinstance(a, torch.Tensor))
                  if s is not None]
        extras += [f"{k}={s}" for k, v in kwargs.items()
                   if (s := arg_sig(v)) is not None and k not in ("device", "layout",
                                                                  "pin_memory")]
        key = (
            str(func.name()).removeprefix("aten::"),
            tuple(tensor_sig(x) for x in ins),
            tuple(extras),
            tuple(tensor_sig(x) for x in outs),
        )
        if name in GATHER_OPS:
            # The source table is mostly untouched: count rows read (== output)
            # plus indices, not the whole embedding matrix.
            nbytes = sum(x.numel() * x.element_size() for x in ins[1:] + outs * 2)
        else:
            nbytes = sum(x.numel() * x.element_size() for x in ins + outs)
        rec = self.sigs.get(key)
        if rec is None:
            tags = func.tags
            rec = self.sigs[key] = {
                "count": 0, "bytes": 0,
                "tag": ("pointwise" if torch.Tag.pointwise in tags else
                        "reduction" if torch.Tag.reduction in tags else None),
            }
        rec["count"] += 1
        rec["bytes"] += nbytes
        self.calls += 1
        return out


# ---------------------------------------------------------------------------
# model zoo
# ---------------------------------------------------------------------------

def _hf_config(repo_id: str):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(repo_id)


def _config_dtype(cfg) -> torch.dtype:
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is None:
            continue
        d = getattr(c, "dtype", None) or getattr(c, "torch_dtype", None)
        if isinstance(d, str):
            d = getattr(torch, d)
        if isinstance(d, torch.dtype):
            return d
    return torch.float32


def _num_layers(cfg) -> int:
    c = getattr(cfg, "text_config", cfg)
    return c.num_hidden_layers


def llm(repo_id: str, *, max_full_layers: int = 40, seq: int = 512, train: bool = False):
    """Prefill + one decode step (+ optionally a training step) of a causal LM.

    Models deeper than `max_full_layers` are traced with 2 layers; every layer
    runs identical shapes, so only the counts change, and the metadata records
    both depths so the renderer can say so.
    """
    def build():
        from transformers import AutoModelForCausalLM
        cfg = _hf_config(repo_id)
        dtype = _config_dtype(cfg)
        full = _num_layers(cfg)
        traced = full if full <= max_full_layers else 2
        getattr(cfg, "text_config", cfg).num_hidden_layers = traced
        with torch.device(DEV):
            model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
        model.eval()
        vocab = getattr(cfg, "text_config", cfg).vocab_size
        ids = torch.randint(0, vocab, (1, seq), device=DEV)
        meta = {"dtype": str(dtype).removeprefix("torch."), "layers_traced": traced,
                "layers_full": full, "input": f"batch 1, {seq} tokens" + ("; train on 256" if train else "")}

        def prefill():
            with torch.no_grad():
                return model(ids, use_cache=True)

        state = {}

        def decode_setup():
            state["pkv"] = prefill().past_key_values

        def decode():
            with torch.no_grad():
                model(ids[:, -1:], past_key_values=state["pkv"], use_cache=True)

        phases = {"prefill": (None, prefill), "decode": (decode_setup, decode)}
        if train:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

            def train_step():
                model.train()
                opt.zero_grad(set_to_none=True)
                model(ids[:, :256], labels=ids[:, :256]).loss.backward()

            phases["train"] = (None, train_step)
            # Traced on its own: the optimizer is pure elementwise traffic over
            # every parameter and would otherwise drown the fwd/bwd mix.
            phases["optimizer"] = (train_step, opt.step)
        return meta, phases
    return build


def hf_encoder(kind: str, dtype=torch.float32, batch: int = 8, seq: int = 128):
    def build():
        import transformers as T
        if kind == "bert-base":
            cfg = T.BertConfig()  # defaults are bert-base-uncased
            model = T.BertModel(cfg)
        elif kind == "t5-small":
            cfg = T.AutoConfig.from_pretrained("t5-small")
            model = T.T5ForConditionalGeneration(cfg)
        elif kind == "gpt2":
            cfg = T.AutoConfig.from_pretrained("distilgpt2")
            model = T.GPT2LMHeadModel(cfg)
        else:
            raise KeyError(kind)
        model = model.to(DEV, dtype).eval()
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=DEV)

        def fwd():
            with torch.no_grad():
                if kind == "t5-small":
                    model(input_ids=ids, decoder_input_ids=ids[:, :32])
                else:
                    model(ids)
        meta = {"dtype": str(dtype).removeprefix("torch."),
                "input": f"batch {batch}, {seq} tokens"}
        return meta, {"forward": (None, fwd)}
    return build


def torchvision_model(name: str, dtype=torch.float32, batch: int = 8, res: int = 224,
                      channels_last: bool = False, train: bool = False):
    def build():
        import torchvision
        model = torchvision.models.get_model(name, weights=None).to(DEV, dtype)
        mf = torch.channels_last if channels_last else torch.contiguous_format
        model = model.to(memory_format=mf)
        x = torch.randn(batch, 3, res, res, device=DEV, dtype=dtype).to(memory_format=mf)
        meta = {"dtype": str(dtype).removeprefix("torch."),
                "input": f"batch {batch}, {res}x{res}" + (", channels_last" if channels_last else "")}

        def fwd():
            model.eval()
            with torch.no_grad():
                model(x)
        phases = {"forward": (None, fwd)}
        if train:
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            y = torch.randint(0, 1000, (batch,), device=DEV)

            def train_step():
                model.train()
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(x), y).backward()

            phases["train"] = (None, train_step)
            phases["optimizer"] = (train_step, opt.step)
        return meta, phases
    return build


def detection(name: str):
    def build():
        import torchvision
        model = torchvision.models.get_model(name, weights=None, weights_backbone=None)
        model = model.to(DEV).eval()
        x = [torch.rand(3, 480, 640, device=DEV)]

        def fwd():
            with torch.no_grad():
                model(x)
        return {"dtype": "float32", "input": "1 image, 480x640"}, {"forward": (None, fwd)}
    return build


def whisper():
    """whisper-small dimensions; fp16 is how it is distributed and usually run."""
    def build():
        import transformers as T
        cfg = T.WhisperConfig(d_model=768, encoder_layers=12, decoder_layers=12,
                              encoder_attention_heads=12, decoder_attention_heads=12,
                              encoder_ffn_dim=3072, decoder_ffn_dim=3072)
        dtype = torch.float16
        model = T.WhisperForConditionalGeneration(cfg).to(DEV, dtype).eval()
        mel = torch.randn(1, 80, 3000, device=DEV, dtype=dtype)
        dec = torch.randint(0, cfg.vocab_size, (1, 4), device=DEV)
        state = {}

        def encode():
            with torch.no_grad():
                state["enc"] = model.model.encoder(mel)

        def decode_setup():
            encode()
            with torch.no_grad():
                out = model(encoder_outputs=state["enc"], decoder_input_ids=dec, use_cache=True)
            state["pkv"] = out.past_key_values

        def decode():
            with torch.no_grad():
                model(encoder_outputs=state["enc"], decoder_input_ids=dec[:, -1:],
                      past_key_values=state["pkv"], use_cache=True)
        meta = {"dtype": "float16", "input": "30 s audio (80x3000 mel)",
                "note": "whisper-small dimensions"}
        return meta, {"encoder": (None, encode), "decode": (decode_setup, decode)}
    return build


def wav2vec2():
    def build():
        import transformers as T
        model = T.Wav2Vec2Model(T.Wav2Vec2Config()).to(DEV).eval()  # base dimensions
        wav = torch.randn(1, 160000, device=DEV)

        def fwd():
            with torch.no_grad():
                model(wav)
        return {"dtype": "float32", "input": "10 s audio at 16 kHz"}, {"forward": (None, fwd)}
    return build


def sd15(part: str):
    """Stable Diffusion 1.5 UNet step / VAE decode, configs from the local cache."""
    def build():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download("sd-legacy/stable-diffusion-v1-5",
                                      allow_patterns=["*/config.json"]))
        dtype = torch.float16
        if part == "unet":
            from diffusers import UNet2DConditionModel
            model = UNet2DConditionModel.from_config(
                UNet2DConditionModel.load_config(root / "unet")).to(DEV, dtype).eval()
            lat = torch.randn(2, 4, 64, 64, device=DEV, dtype=dtype)
            ctx = torch.randn(2, 77, 768, device=DEV, dtype=dtype)
            t = torch.tensor([500, 500], device=DEV)

            def fwd():
                with torch.no_grad():
                    model(lat, t, encoder_hidden_states=ctx)
            meta = {"dtype": "float16", "input": "512x512, CFG batch 2 (latent 2x4x64x64)"}
        else:
            from diffusers import AutoencoderKL
            model = AutoencoderKL.from_config(
                AutoencoderKL.load_config(root / "vae")).to(DEV, dtype).eval()
            lat = torch.randn(1, 4, 64, 64, device=DEV, dtype=dtype)

            def fwd():
                with torch.no_grad():
                    model.decode(lat)
            meta = {"dtype": "float16", "input": "latent 1x4x64x64 -> 512x512 image"}
        return meta, {"forward": (None, fwd)}
    return build


def _diffusers_fwd(make, dtype=torch.bfloat16, note=""):
    """Wrap `make() -> (model, kwargs, meta_input)` into a single-forward builder."""
    def build():
        model, kwargs, input_desc = make()
        model = model.to(DEV, dtype).eval()
        kwargs = {k: (v.to(DEV, dtype) if isinstance(v, torch.Tensor) and v.is_floating_point()
                      else v.to(DEV) if isinstance(v, torch.Tensor) else v)
                  for k, v in kwargs.items()}

        def fwd():
            with torch.no_grad():
                model(**kwargs)
        meta = {"dtype": str(dtype).removeprefix("torch."), "input": input_desc}
        if note:
            meta["note"] = note
        return meta, {"forward": (None, fwd)}
    return build


def _vae_decode(make, latent_shape, dtype=torch.bfloat16, note=""):
    def build():
        model = make().to(DEV, dtype).eval()
        z = torch.randn(*latent_shape, device=DEV, dtype=dtype)

        def fwd():
            with torch.no_grad():
                model.decode(z)
        meta = {"dtype": str(dtype).removeprefix("torch."),
                "input": "latent " + "x".join(map(str, latent_shape))}
        if note:
            meta["note"] = note
        return meta, {"forward": (None, fwd)}
    return build


def _sdxl_unet():
    from diffusers import UNet2DConditionModel
    model = UNet2DConditionModel(
        sample_size=128, cross_attention_dim=2048, block_out_channels=(320, 640, 1280),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
        transformer_layers_per_block=(1, 2, 10), attention_head_dim=(5, 10, 20),
        addition_embed_type="text_time", addition_time_embed_dim=256,
        projection_class_embeddings_input_dim=2816, use_linear_projection=True,
        layers_per_block=2)
    kw = {"sample": torch.randn(2, 4, 128, 128), "timestep": torch.tensor([500, 500]),
          "encoder_hidden_states": torch.randn(2, 77, 2048),
          "added_cond_kwargs": {"text_embeds": torch.randn(2, 1280).to(DEV, torch.float16),
                                "time_ids": torch.randn(2, 6).to(DEV, torch.float16)}}
    return model, kw, "1024x1024, CFG batch 2 (latent 2x4x128x128)"


def _sd_vae():
    from diffusers import AutoencoderKL
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("sd-legacy/stable-diffusion-v1-5", allow_patterns=["*/config.json"]))
    return AutoencoderKL.from_config(AutoencoderKL.load_config(root / "vae"))


def _flux_config(sub):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("black-forest-labs/FLUX.1-dev", allow_patterns=["*/config.json"]))
    return root / sub


def _flux_transformer():
    from diffusers import FluxTransformer2DModel
    cfg = FluxTransformer2DModel.load_config(_flux_config("transformer"))
    cfg.update(num_layers=1, num_single_layers=2)
    model = FluxTransformer2DModel.from_config(cfg)
    kw = {"hidden_states": torch.randn(1, 4096, 64), "encoder_hidden_states": torch.randn(1, 512, 4096),
          "pooled_projections": torch.randn(1, 768), "timestep": torch.tensor([0.5]),
          "img_ids": torch.zeros(4096, 3), "txt_ids": torch.zeros(512, 3),
          "guidance": torch.tensor([3.5])}
    return model, kw, "1024x1024 (4096 image + 512 text tokens)"


def _flux_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_config(AutoencoderKL.load_config(_flux_config("vae")))


def _sd35_transformer():
    from diffusers import SD3Transformer2DModel
    model = SD3Transformer2DModel(
        sample_size=128, patch_size=2, in_channels=16, num_layers=2, attention_head_dim=64,
        num_attention_heads=38, joint_attention_dim=4096, caption_projection_dim=2432,
        pooled_projection_dim=2048, out_channels=16, pos_embed_max_size=192, qk_norm="rms_norm")
    kw = {"hidden_states": torch.randn(2, 16, 128, 128), "encoder_hidden_states": torch.randn(2, 333, 4096),
          "pooled_projections": torch.randn(2, 2048), "timestep": torch.tensor([500, 500])}
    return model, kw, "1024x1024, CFG batch 2 (4096 image + 333 text tokens)"


def _qwen_image_transformer():
    from diffusers import QwenImageTransformer2DModel
    model = QwenImageTransformer2DModel(num_layers=2)
    txt = 128
    kw = {"hidden_states": torch.randn(1, 4096, 64), "encoder_hidden_states": torch.randn(1, txt, 3584),
          "encoder_hidden_states_mask": torch.ones(1, txt, dtype=torch.long),
          "timestep": torch.tensor([0.5]), "img_shapes": [(1, 64, 64)], "txt_seq_lens": [txt]}
    return model, kw, "1024x1024 (4096 image + 128 text tokens)"


def _ltx_transformer():
    from diffusers import LTXVideoTransformer3DModel
    model = LTXVideoTransformer3DModel(num_layers=2)
    f, h, w = 5, 16, 24   # 33 frames at 512x768 -> 1920 tokens
    kw = {"hidden_states": torch.randn(1, f * h * w, 128), "encoder_hidden_states": torch.randn(1, 128, 4096),
          "timestep": torch.tensor([500]), "encoder_attention_mask": torch.ones(1, 128, dtype=torch.long),
          "num_frames": f, "height": h, "width": w}
    return model, kw, "33 frames at 512x768 (1920 video + 128 text tokens)"


def _ltx_vae():
    from diffusers import AutoencoderKLLTXVideo
    return AutoencoderKLLTXVideo()


def _wan_transformer():
    from diffusers import WanTransformer3DModel
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=12, attention_head_dim=128, in_channels=16,
        out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=8960, num_layers=2,
        cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6)
    kw = {"hidden_states": torch.randn(1, 16, 9, 40, 60), "encoder_hidden_states": torch.randn(1, 512, 4096),
          "timestep": torch.tensor([500])}
    return model, kw, "33 frames at 320x480 (5400 video + 512 text tokens)"


def _wan_vae():
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan()


def _hf_vision(kind):
    def build():
        import transformers as T
        if kind == "zoedepth":
            model, x = T.ZoeDepthForDepthEstimation(T.ZoeDepthConfig()), torch.randn(1, 3, 384, 512)
        elif kind == "depth-anything":
            model, x = T.DepthAnythingForDepthEstimation(T.DepthAnythingConfig()), torch.randn(1, 3, 518, 518)
        elif kind == "segformer":
            model = T.SegformerForSemanticSegmentation(T.SegformerConfig(num_labels=150))
            x = torch.randn(1, 3, 512, 512)
        else:
            raise KeyError(kind)
        model = model.to(DEV).eval()
        x = x.to(DEV)

        def fwd():
            with torch.no_grad():
                model(pixel_values=x)
        return {"dtype": "float32", "input": "x".join(map(str, x.shape[1:])),
                "note": "ComfyUI ControlNet preprocessor class"}, {"forward": (None, fwd)}
    return build


# name -> (family, builder). Families group models in WORKLOADS.md.
MODELS = {
    "llama-3.2-1b": ("llm", llm("meta-llama/Llama-3.2-1B", train=True)),
    "llama-3.1-8b": ("llm", llm("meta-llama/Meta-Llama-3.1-8B-Instruct", max_full_layers=0)),
    "qwen3-0.6b": ("llm", llm("Qwen/Qwen3-0.6B")),
    "gemma-2-2b": ("llm", llm("google/gemma-2-2b-it")),
    "phi-2": ("llm", llm("microsoft/phi-2")),
    "bert-base": ("encoder", hf_encoder("bert-base")),
    "t5-small": ("encoder", hf_encoder("t5-small", seq=512, batch=1)),
    "gpt2": ("encoder", hf_encoder("gpt2", seq=512, batch=1)),
    "resnet50": ("vision", torchvision_model("resnet50", train=True)),
    "resnet50-f16-cl": ("vision", torchvision_model("resnet50", dtype=torch.float16,
                                                     channels_last=True)),
    "mobilenet_v3_large": ("vision", torchvision_model("mobilenet_v3_large")),
    "efficientnet_b0": ("vision", torchvision_model("efficientnet_b0")),
    "convnext_tiny": ("vision", torchvision_model("convnext_tiny")),
    "swin_t": ("vision", torchvision_model("swin_t")),
    "vit_b_16": ("vision", torchvision_model("vit_b_16")),
    "fasterrcnn_mobilenet": ("vision", detection("fasterrcnn_mobilenet_v3_large_fpn")),
    "whisper-small": ("audio", whisper()),
    "wav2vec2-base": ("audio", wav2vec2()),
    "sd15-unet": ("diffusion", sd15("unet")),
    "sd15-vae-decode": ("diffusion", sd15("vae")),
    # The models ComfyUI users cite in module: mps issues (#139389, #155797,
    # #141471, #187280, #194922). Big transformers run a couple of blocks.
    "sdxl-unet": ("diffusion", _diffusers_fwd(_sdxl_unet, torch.float16)),
    "sdxl-vae-decode": ("diffusion", _vae_decode(_sd_vae, (1, 4, 128, 128))),
    "flux-dev": ("diffusion", _diffusers_fwd(
        _flux_transformer, note="1 of 19 double + 2 of 38 single blocks")),
    "flux-vae-decode": ("diffusion", _vae_decode(_flux_vae, (1, 16, 128, 128))),
    "sd3.5-large": ("diffusion", _diffusers_fwd(_sd35_transformer, note="2 of 38 layers")),
    "qwen-image": ("diffusion", _diffusers_fwd(_qwen_image_transformer, note="2 of 60 layers")),
    "ltx-video": ("video", _diffusers_fwd(_ltx_transformer, note="2 of 28 layers")),
    "ltx-video-vae-decode": ("video", _vae_decode(_ltx_vae, (1, 128, 5, 16, 24))),
    "wan2.1-1.3b": ("video", _diffusers_fwd(_wan_transformer, note="2 of 30 layers")),
    "wan-vae-decode": ("video", _vae_decode(_wan_vae, (1, 16, 3, 40, 60))),
    "zoedepth": ("preprocessor", _hf_vision("zoedepth")),
    "depth-anything": ("preprocessor", _hf_vision("depth-anything")),
    "segformer-ade": ("preprocessor", _hf_vision("segformer")),
}


def trace_one(name: str) -> dict:
    family, builder = MODELS[name]
    torch.manual_seed(0)
    meta, phases = builder()
    result = {"model": name, "family": family, **meta,
              "torch": torch.__version__, "phases": {}}
    for phase, (setup, fn) in phases.items():
        if setup is not None:
            setup()
        fn()  # warm up: first calls take one-off paths (caches, lazy init)
        torch.mps.synchronize()
        if setup is not None:
            setup()
        rec = OpRecorder()
        t0 = time.perf_counter()
        with rec:
            fn()
        torch.mps.synchronize()
        ops = [
            {"op": k[0], "inputs": list(k[1]), "args": list(k[2]), "outputs": list(k[3]), **v}
            for k, v in sorted(rec.sigs.items(), key=lambda kv: -kv[1]["bytes"])
        ]
        result["phases"][phase] = {"calls": rec.calls, "wall_s": time.perf_counter() - t0,
                                   "ops": ops}
        print(f"[trace] {name}/{phase}: {rec.calls} calls, {len(ops)} signatures",
              file=sys.stderr)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trace_models")
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)
    if args.list:
        for n, (fam, _) in MODELS.items():
            print(f"{fam:10s} {n}")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in args.models:
        try:
            res = trace_one(name)
        except Exception as e:  # keep going: one broken import should not cost the zoo
            print(f"[trace] !! {name}: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append(name)
            continue
        finally:
            torch.mps.empty_cache()
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=1))
    if failures:
        print(f"[trace] failed: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
