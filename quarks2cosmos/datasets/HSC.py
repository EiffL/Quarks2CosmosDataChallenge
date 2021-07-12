""" TensorFlow Dataset of HSC images. """
import os
from astropy.table import Table
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import h5py

_CITATION = """
"""

_DESCRIPTION = """
"""

class HSC(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for HSC dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain a tar file with images (HSC_dataset.tar.gz).
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(kappatng): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image": tfds.features.Tensor(shape=[41, 41], dtype=tf.float32),
            "psf":   tfds.features.Tensor(shape=[41, 41], dtype=tf.float32),
            "variance": tfds.features.Tensor(shape=[41, 41], dtype=tf.float32),
            "mask": tfds.features.Tensor(shape=[41, 41], dtype=tf.int32),
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
    data_path = dl_manager.extract(os.path.join(dl_manager.manual_dir, "HSC_dataset.tar.gz")) 

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "catalog_filename": os.path.join(data_path, "catalog_masked_obj.fits"),
                "cutouts_filename": os.path.join(data_path, "cutouts_pdr2_wide_coadd.hdf"),
                "psfs_filename": os.path.join(data_path, "psfs_pdr2_wide_coadd.hdf"),
            },)
    ]

  def _generate_examples(self, catalog_filename, cutouts_filename, psfs_filename):
    """Yields examples."""
    # Load the catalog
    catalog = Table.read(catalog_filename)

    with h5py.File(cutouts_filename,'r') as cutouts, h5py.File(psfs_filename,'r') as psfs:
        # Loop through the examples, resize cutout to desired size
        for row in catalog:
            try:
                obj_id = str(row['object_id'])
                cutout = cutouts[obj_id]['HSC-I']['HDU0']['DATA'][:].astype('float32')
                mask = cutouts[obj_id]['HSC-I']['HDU1']['DATA'][:].astype('int32')
                variance = cutouts[obj_id]['HSC-I']['HDU2']['DATA'][:].astype('float32')
                psf = psfs[obj_id]['HSC-I']['PRIMARY']['DATA'][:].astype('float32')

                # Making sure stamps have desired size
                cutout = tf.image.resize_with_crop_or_pad(cutout[...,np.newaxis], 41, 41)[...,0].numpy()
                mask = tf.image.resize_with_crop_or_pad(mask[...,np.newaxis], 41, 41)[...,0].numpy()
                variance = tf.image.resize_with_crop_or_pad(variance[...,np.newaxis], 41, 41)[...,0].numpy()
                psf = tf.image.resize_with_crop_or_pad(psf[...,np.newaxis], 41, 41)[...,0].numpy()

                yield obj_id, {"image": cutout,
                                "psf":psf,
                                "variance": variance,
                                "mask":  mask}
            except KeyError:
                print('Dataset missing key', str(row['object_id']))
