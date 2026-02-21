"""Unit tests for the Diffusers-backed StabilityVAE wrapper."""

# built-in libs
import unittest
from types import SimpleNamespace

# external libs
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

# deps
from networks.encoders.sd_vae import StabilityVAE


class _FakePosterior:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, key):
        del key
        return self.mean + 1.0


class _FakeVAE:
    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=0.18215)

    def encode(self, x):
        del x

    def decode(self, z):
        del z

    def apply(self, variables, x, method):
        params = variables["params"]
        if method == self.encode:
            b, _, h, w = x.shape
            mean = jnp.zeros((b, 4, h // 8, w // 8), dtype=jnp.float32)
            return SimpleNamespace(latent_dist=_FakePosterior(mean))

        if method == self.decode:
            b, _, h, w = x.shape
            # Decoded image range is expected in [-1, 1] before uint8 conversion.
            decoded = jnp.zeros((b, 3, h * 8, w * 8), dtype=jnp.float32)
            return SimpleNamespace(sample=decoded, params_seen=params)

        raise ValueError("Unexpected method passed to fake VAE")


def _build_test_encoder() -> StabilityVAE:
    encoder = StabilityVAE.__new__(StabilityVAE)
    encoder.config = ml_collections.ConfigDict(dict(latent_channels=4))
    encoder.dtype = jnp.float32
    encoder.encoded_pixels = False
    encoder.rngs = None
    encoder.module = _FakeVAE()
    encoder.params_host = {"w": np.array([1.0], dtype=np.float32)}
    encoder.params_tpu = None
    encoder._params_tpu_cache_key = None
    encoder.scaling_factor = float(encoder.module.config.scaling_factor)
    return encoder


class StabilityVAETest(unittest.TestCase):
    def test_encode_decode_cpu_shapes_and_dtypes(self):
        encoder = _build_test_encoder()
        images = jnp.zeros((2, 256, 256, 3), dtype=jnp.float32)
        latents = encoder.encode(images, sample_posterior=False)
        decoded = encoder.decode(latents)

        self.assertEqual(latents.shape, (2, 32, 32, 4))
        self.assertEqual(decoded.shape, (2, 256, 256, 3))
        self.assertEqual(decoded.dtype, jnp.uint8)

    def test_get_params_for_input_cpu_returns_host(self):
        encoder = _build_test_encoder()
        x = jnp.zeros((1, 32, 32, 4), dtype=jnp.float32)
        params = encoder._get_params_for_input(x)
        self.assertIs(params, encoder.params_host)

    def test_encoded_pixels_bypass(self):
        encoder = _build_test_encoder()
        encoder.encoded_pixels = True
        latents = jnp.ones((1, 32, 32, 4), dtype=jnp.float32)
        encoded = encoder.encode(latents)
        self.assertEqual(encoded.shape, latents.shape)
        self.assertEqual(encoded.dtype, jnp.float32)

    def test_tpu_params_cache(self):
        tpu_devices = [d for d in jax.devices() if d.platform == "tpu"]
        if not tpu_devices:
            self.skipTest("No TPU device available.")

        encoder = _build_test_encoder()
        x = jax.device_put(jnp.zeros((1, 32, 32, 4), dtype=jnp.float32), tpu_devices[0])

        params1 = encoder._get_params_for_input(x)
        params2 = encoder._get_params_for_input(x)
        self.assertIs(params1, params2)

        leaf = jax.tree_util.tree_leaves(params1)[0]
        self.assertTrue(any(d.platform == "tpu" for d in leaf.devices()))


if __name__ == "__main__":
    unittest.main()
