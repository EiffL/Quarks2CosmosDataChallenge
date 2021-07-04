""" TensorFlow Dataset of COSMOS images. """
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import galsim as gs

_CITATION = """
"""

_DESCRIPTION = """
"""

class CosmosConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Cosmos."""

  def __init__(self, *, sample="23.5", stamp_size=101, pixel_scale=0.03, **kwargs):
    """BuilderConfig for Cosmos.
    Args:
      sample: which Cosmos sample to use, "23.5" or "25.2".
      stamp_size: image stamp size in pixels.
      pixel_scale: pixel scale of stamps in arcsec.
      **kwargs: keyword arguments forwarded to super.
    """
    v1 = tfds.core.Version("0.0.1")
    super(CosmosConfig, self).__init__(
        description=("Cosmos stamps from %s sample in %d x %d resolution, %.2f arcsec/pixel." %
                      (sample, stamp_size, stamp_size, pixel_scale)),
        version=v1,
        **kwargs)
    self.stamp_size = stamp_size
    self.pixel_scale = pixel_scale
    self.sample = sample


class Cosmos(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Cosmos dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }
  
  BUILDER_CONFIGS = [CosmosConfig(name="Cosmos_23.5", sample="23.5"),
                     CosmosConfig(name="Cosmos_25.2", sample="25.2")]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                 self.builder_config.stamp_size], dtype=tf.float32),
            "psf":   tfds.features.Tensor(shape=[self.builder_config.stamp_size,
                                                 self.builder_config.stamp_size], dtype=tf.float32)
	}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("image", "image"),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
            "offset": 0,
            "size": 40000,
            },),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
            "offset": 40000,
            "size": 10000,
            },
        ),
    ]

  def _generate_examples(self, offset, size):
    """Yields examples."""
    # Loads the galsim COSMOS catalog
    cat = gs.COSMOSCatalog(sample=self.builder_config.sample)
    ngal = size #cat.getNObjects()

    #yield 'key', {}
    for i in range(ngal):
      gal = cat.makeGalaxy(i+offset)
      cosmos_gal = gs.Convolve(gal, gal.original_psf)
      
      cosmos_stamp = cosmos_gal.drawImage(nx=self.builder_config.stamp_size, 
                                          ny=self.builder_config.stamp_size, 
                                          scale=self.builder_config.pixel_scale, 
                                          method='no_pixel').array.astype('float32')
                                          
      cosmos_psf_stamp = gal.original_psf.drawImage(nx=self.builder_config.stamp_size, 
                                          ny=self.builder_config.stamp_size, 
                                          scale=self.builder_config.pixel_scale,
                                          method='no_pixel').array.astype('float32')
      yield '%d'%i, {"image": cosmos_stamp, "psf":cosmos_psf_stamp}