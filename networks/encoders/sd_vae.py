"""Diffusers-backed Stability VAE encoder."""

from __future__ import annotations

# external libs
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections
from diffusers import FlaxAutoencoderKL


class StabilityVAE(nnx.Module):
    """Wrapper around `FlaxAutoencoderKL` for latent diffusion training."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        dtype: jnp.dtype = jnp.float32,
        encoded_pixels: bool = False,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.encoded_pixels = encoded_pixels
        self.rngs = rngs

        ckpt = config.get("pretrained_path", None) or "pcuenq/sd-vae-ft-mse-flax"
        module, params = FlaxAutoencoderKL.from_pretrained(ckpt)
        self.module = module
        self.params = params
        self.scaling_factor = float(module.config.scaling_factor)

    @staticmethod
    def _to_nchw(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(x, (0, 3, 1, 2))

    @staticmethod
    def _to_nhwc(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.transpose(x, (0, 2, 3, 1))

    def _sample_key(self):
        if self.rngs is None:
            return jax.random.PRNGKey(0)
        return self.rngs()

    @nnx.jit
    def encode(self, x: jnp.ndarray, sample_posterior: bool = True, deterministic: bool = True) -> jnp.ndarray:
        # `latent_dataset=True` path: input is already latent, so skip VAE encode.
        if self.encoded_pixels:
            return x.astype(self.dtype)

        del deterministic  # No dropout path in this encoder wrapper.

        # Optional convenience path when input is uint8 pixels.
        if x.dtype == jnp.uint8:
            x = x.astype(jnp.float32) / 127.5 - 1.0
        x = x.astype(self.dtype)

        x_nchw = self._to_nchw(x)
        posterior = self.module.apply(
            {"params": self.params},
            x_nchw,
            method=self.module.encode,
        ).latent_dist

        if sample_posterior:
            z = posterior.sample(self._sample_key())
        else:
            z = posterior.mean

        z = self._to_nhwc(z)
        return z * self.scaling_factor

    @nnx.jit
    def decode(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        del deterministic  # No dropout path in this encoder wrapper.
        z = z.astype(self.dtype) / self.scaling_factor
        z_nchw = self._to_nchw(z)

        x = self.module.apply(
            {"params": self.params},
            z_nchw,
            method=self.module.decode,
        ).sample
        x = self._to_nhwc(x)
        return (x.astype(jnp.float32) * 127.5 + 128.0).clip(0, 255).astype(jnp.uint8)

    def load_pretrained(self, pretrained_path: str):
        module, params = FlaxAutoencoderKL.from_pretrained(pretrained_path)
        self.module = module
        self.params = params
        self.scaling_factor = float(module.config.scaling_factor)
