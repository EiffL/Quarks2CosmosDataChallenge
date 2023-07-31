import jax
import jax.numpy as jnp
import haiku as hk
import numpy as onp
from quarks2cosmos.models.convdae import BlockGroup
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class Encoder(hk.Module):
  def __init__(self,
               latent_size,
               blocks_per_group=(2, 2, 2, 2, 2),
               bn_config=False,
               bottleneck=False,
               channels_per_group=(32, 64, 128, 128, 32),
               use_projection=(True, True, True, True, True),
               name=None):
    """Constructs a Residual UNet model based on a traditional ResNet.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      name: Name of the module.
    """
    super().__init__(name=name)

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    self.bn_config = bn_config
    self.channels_per_group = channels_per_group
    self.blocks_per_group = blocks_per_group
    self.use_projection = use_projection
    self.bottleneck = bottleneck
    self.latent_size = latent_size
    
  def __call__(self, inputs, is_training=False, test_local_stats=False):
    
    # Adding channel dimension
    x = inputs[..., jnp.newaxis]
    
    net = hk.Conv2D(
        output_channels=32,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding="SAME",
        name="initial_conv")(x)

    strides = (1, 2, 2, 2, 2)
    for i in range(5):
      net = BlockGroup(channels=self.channels_per_group[i],
                     num_blocks=self.blocks_per_group[i],
                     stride=strides[i],
                     bn_config=self.bn_config,
                     bottleneck=self.bottleneck,
                     use_projection=self.use_projection[i],
                     transpose=False,
                     name="block_group_%d" % (i))(net, is_training, test_local_stats)
    
    # Bottleneck layer
    net = hk.Flatten()(net) # This should be of size 512
    loc = hk.Linear(self.latent_size)(net)
    scale = jax.nn.softplus(hk.Linear(self.latent_size)(net)) + 1e-3
    
    return tfd.MultivariateNormalDiag(loc, scale) 


class Decoder(hk.Module):
  def __init__(self,
               blocks_per_group=(2, 2, 2, 2, 2),
               bn_config=False,
               bottleneck=False,
               channels_per_group=(32, 64, 128, 128, 32),
               use_projection=(True, True, True, True, True),
               name=None):
    """Constructs a Residual UNet model based on a traditional ResNet.
    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
      bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
      name: Name of the module.
    """
    super().__init__(name=name)

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    self.bn_config = bn_config
    self.channels_per_group = channels_per_group
    self.blocks_per_group = blocks_per_group
    self.use_projection = use_projection
    self.bottleneck = bottleneck
    
  def __call__(self, inputs, is_training=False, test_local_stats=False):
    
    # Expand and reshape inputs
    net = hk.Linear(3*3*32)(inputs)
    net = net.reshape([-1,3,3,32])
    
    strides = (1, 2, 2, 2, 2)
    for i in onp.arange(5)[::-1]:
      net = BlockGroup(channels=self.channels_per_group[i],
                     num_blocks=self.blocks_per_group[i],
                     stride=strides[i],
                     bn_config=self.bn_config,
                     bottleneck=self.bottleneck,
                     use_projection=self.use_projection[i],
                     transpose=True,
                     name="up_block_group_%d" % (i))(net, is_training, test_local_stats)
    
    net = hk.Conv2DTranspose(output_channels=1,
                                kernel_shape=5,
                                stride=2,
                                padding="SAME",
                                name="final_conv")(net)
    
    # Adding padding to recover 101x101 shape
    net = net[...,0]
    net = jnp.pad(net,[[0,0],[3,2],[3,2]])
    return tfd.Independent(tfd.MultivariateNormalDiag(net),
                           reinterpreted_batch_ndims=1)
