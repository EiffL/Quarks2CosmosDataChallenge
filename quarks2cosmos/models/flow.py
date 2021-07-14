import jax
import jax.numpy as jnp
import haiku as hk

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class AffineCoupling(hk.Module):
  """This is the coupling layer used in the Flow."""
  def __init__(self, scale_only=True, **kwargs):
    super().__init__(**kwargs)
    self.scale_only = scale_only

  def __call__(self, x, output_units, **condition_kwargs):
    net = hk.Linear(128)(x)
    net = jax.nn.leaky_relu(net)
    net = hk.Linear(128)(net)
    net = jax.nn.leaky_relu(net)
    shifter = tfb.Shift(hk.Linear(output_units)(net))
    if self.scale_only:
      return shifter
    else:
      scaler = tfb.Scale(jnp.clip(jax.nn.softplus(hk.Linear(output_units)(net)), 1e-2, 1e1))
      return tfb.Chain([shifter, scaler])


class AffineFlow(hk.Module):
    """This is a normalizing flow using the coupling layers defined
    above."""
    def __init__(self, d=32,name=None ):
        super().__init__(name=name)
        self.d=d

    def __call__(self):
        chain = tfb.Chain([
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff1')),
            tfb.Permute(jnp.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff2')),
            tfb.Permute(jnp.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff3',
                                                         scale_only=False)),
            tfb.Permute(jnp.arange(self.d)[::-1]),
            tfb.RealNVP(self.d//2, bijector_fn=AffineCoupling(name='aff4',
                                                         scale_only=False)),
            tfb.Permute(jnp.arange(self.d)[::-1]),
        ])
        
        nvp = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(jnp.zeros(self.d),jnp.ones(self.d)),
            bijector=chain)
        return nvp