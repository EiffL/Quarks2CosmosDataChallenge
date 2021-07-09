import haiku as hk
import jax
import jax.numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

class Encoder(hk.Module):
  """Simple Convolutional encoder model."""
  def __init__(self, latent_size=32):
    super().__init__()
    self._latent_size = latent_size

  def __call__(self, x):
    x = hk.Conv2D(16, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(32, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x,  window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(64, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(128, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Flatten()(x)
    
    # Returns the variational distribution encoding the input image
    loc = hk.Linear(self._latent_size)(x)
    scale = jax.nn.softplus(hk.Linear(self._latent_size)(x) + 1e-5)
    return tfd.MultivariateNormalDiag(loc, scale)

class Decoder(hk.Module):
  """Simple Convolutional decoder model."""
  def __call__(self, z, scale=0.01):
    
    # Reshape latent variable to an image
    x = hk.Linear(3*3*128)(z)
    x = x.reshape([-1,3,3,128])
    
    x = hk.Conv2DTranspose(64, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(32, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(16, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(1, kernel_shape=3, stride=2)(x)
    x = jnp.pad(x, [[0,0],[1,2],[1,2],[0,0]])
    
    return tfd.Independent(tfd.MultivariateNormalDiag(x, 
                           scale_identity_multiplier=scale),
                           reinterpreted_batch_ndims=2)
