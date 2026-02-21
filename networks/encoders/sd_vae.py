"""Diffusers-backed Stability VAE encoder."""

from __future__ import annotations

# external libs
from typing import Any
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
try:
    from diffusers import FlaxAutoencoderKL
except ModuleNotFoundError:
    FlaxAutoencoderKL = None


class StabilityVAE:
    """Wrapper around `FlaxAutoencoderKL` for latent diffusion training."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        dtype: jnp.dtype = jnp.float32,
        encoded_pixels: bool = False,
        rngs: Any | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.encoded_pixels = encoded_pixels
        self.rngs = rngs

        if FlaxAutoencoderKL is None:
            raise ImportError("diffusers is required for StabilityVAE. Please install `diffusers`.")

        ckpt = config.get("pretrained_path", None) or "pcuenq/sd-vae-ft-mse-flax"
        module, params = FlaxAutoencoderKL.from_pretrained(ckpt)
        self.module = module
        self.params_host = jax.device_get(params)
        self.params_tpu = None
        self._params_tpu_cache_key = None
        self.scaling_factor = float(module.config.scaling_factor)

    @staticmethod
    def _to_nchw(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(x, (0, 3, 1, 2))

    @staticmethod
    def _to_nhwc(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(x, (0, 2, 3, 1))

    @staticmethod
    def _is_tpu_array(x: jnp.ndarray) -> bool:
        try:
            return any(device.platform == "tpu" for device in x.devices())
        except Exception:
            pass
        try:
            return x.device().platform == "tpu"
        except Exception:
            return False

    @staticmethod
    def _sharding_cache_key(x: jnp.ndarray):
        sharding = getattr(x, "sharding", None)
        if isinstance(sharding, NamedSharding):
            device_ids = tuple(int(d.id) for d in sharding.mesh.devices.flat)
            return ("named", device_ids)
        if sharding is not None:
            try:
                device_ids = tuple(sorted(int(d.id) for d in sharding.device_set))
                return (type(sharding).__name__, device_ids)
            except Exception:
                return (type(sharding).__name__, repr(sharding))
        try:
            device_ids = tuple(sorted(int(d.id) for d in x.devices()))
            return ("devices", device_ids)
        except Exception:
            return ("unknown",)

    @staticmethod
    def _place_replicated(params, target: jnp.ndarray):
        sharding = getattr(target, "sharding", None)
        if isinstance(sharding, NamedSharding):
            rep_sharding = NamedSharding(sharding.mesh, PartitionSpec())
            return jax.tree.map(lambda v: jax.device_put(jnp.asarray(v), rep_sharding), params)

        try:
            devices = [d for d in target.devices() if d.platform == "tpu"]
        except Exception:
            devices = []

        if not devices:
            devices = [d for d in jax.devices() if d.platform == "tpu"]
        if not devices:
            raise RuntimeError("TPU params requested but no TPU devices are available.")

        device = devices[0]
        return jax.tree.map(lambda v: jax.device_put(jnp.asarray(v), device), params)

    def _get_tpu_params(self, x: jnp.ndarray):
        cache_key = self._sharding_cache_key(x)
        if self.params_tpu is None or self._params_tpu_cache_key != cache_key:
            self.params_tpu = self._place_replicated(self.params_host, x)
            self._params_tpu_cache_key = cache_key
        return self.params_tpu

    def _get_params_for_input(self, x: jnp.ndarray):
        if self._is_tpu_array(x):
            return self._get_tpu_params(x)
        return self.params_host

    def _sample_key(self):
        if self.rngs is None:
            return jax.random.PRNGKey(0)
        try:
            return self.rngs()
        except Exception:
            return jax.random.PRNGKey(0)

    def encode(self, x: jnp.ndarray, sample_posterior: bool = True, deterministic: bool = True) -> jnp.ndarray:
        # `latent_dataset=True` path: input is already latent, so skip VAE encode.
        if self.encoded_pixels:
            return x.astype(self.dtype)

        del deterministic  # No dropout path in this encoder wrapper.
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected NHWC images with 3 channels, got shape {x.shape}.")

        # Optional convenience path when input is uint8 pixels.
        if x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 127.5 - 1.0
        x = x.astype(self.dtype)

        x_nchw = self._to_nchw(x)
        params = self._get_params_for_input(x_nchw)
        posterior = self.module.apply(
            {"params": params},
            x_nchw,
            method=self.module.encode,
        ).latent_dist

        if sample_posterior:
            z = posterior.sample(self._sample_key())
        else:
            z = posterior.mean

        z = self._to_nhwc(z)
        return z * self.scaling_factor

    def decode(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        del deterministic  # No dropout path in this encoder wrapper.
        latent_channels = int(self.config.get("latent_channels", 4))
        if z.ndim != 4 or z.shape[-1] != latent_channels:
            raise ValueError(f"Expected NHWC latents with {latent_channels} channels, got shape {z.shape}.")

        z = z.astype(self.dtype) / self.scaling_factor
        z_nchw = self._to_nchw(z)
        params = self._get_params_for_input(z_nchw)

        x = self.module.apply(
            {"params": params},
            z_nchw,
            method=self.module.decode,
        ).sample
        x = self._to_nhwc(x)
        return (x.astype(jnp.float32) * 127.5 + 128.0).clip(0, 255).astype(jnp.uint8)

    def load_pretrained(self, pretrained_path: str):
        if FlaxAutoencoderKL is None:
            raise ImportError("diffusers is required for StabilityVAE. Please install `diffusers`.")
        module, params = FlaxAutoencoderKL.from_pretrained(pretrained_path)
        self.module = module
        self.params_host = jax.device_get(params)
        self.params_tpu = None
        self._params_tpu_cache_key = None
        self.scaling_factor = float(module.config.scaling_factor)
