# This module contains some simple functions to handle
# common operations on galaxy images, like convolutions
import numpy as onp
import jax.numpy as jnp
import jax

from jax.image import ResizeMethod


def deconvolve(image, 
               psf, 
               return_Fourier=True):
  """ Deconvolve input image from PSF

  Args: 
    image: a JAX array of size [nx, ny].
    psf: a JAX array, must have same shape as image.
    return_Fourier: whether to return a Fourier space image (default: True)
  Returns:
    The deconvolved image, either in real or Fourier space.
  """
  nx,ny = image.shape

  # Compute Fourier transform of input images
  imk = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(image)))
  imkpsf = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(psf)))
  
  # Computing deconvolved galaxy image
  kx, ky = jnp.meshgrid(jnp.fft.fftfreq(nx), jnp.fft.fftfreq(ny))
  mask = jnp.sqrt(kx**2 + ky**2) <= 0.5
  mask = jnp.fft.fftshift(mask)
  im_deconv = imk * ((1./(imkpsf+1e-10))*mask)

  if return_Fourier:
    return im_deconv
  else:
    return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(im_deconv)).real)


def kresample(kimage, 
              input_pixel_scale,
              target_pixel_scale,
              target_npix,
              interpolant=ResizeMethod.LANCZOS5):
  """ Resamples a kimage to target pixel size and image size.
  
  Args:
    kimage: Fourier image to resample, assumes 0 frequency at the center.
    input_pixel_scale: input pixel size in arcsec.
    target_pixel_scale: target pixel size in arcsec.
    target_npix: Size of target image in pixels.
    interpolant: The resizing method to use; either a ``ResizeMethod`` instance or a
      string. Available methods are: LINEAR, LANCZOS3, LANCZOS5, CUBIC.
  Returns:
    The resampled kimage.
  """
  nx, ny = kimage.shape
  stepk = 2.*jnp.pi/(nx * input_pixel_scale)
  target_stepk = 2.*jnp.pi/(target_npix * target_pixel_scale)

  kimage_resamp = jax.image.scale_and_translate(kimage, 
                              shape=[target_npix,target_npix], 
                              spatial_dims=[0,1], 
                              scale=jnp.array([1/(target_stepk/stepk)]*2), 
                              translation=-jnp.array([(nx/2)/(target_stepk/stepk) - target_npix/2]*2), 
                              method=interpolant,
                              antialias=False)
  return kimage_resamp


def convolve(image, psf, return_Fourier=False):
  """ Convolves given image by psf.
  Args: 
    image: a JAX array of size [nx, ny], either in real or Fourier space.
    psf: a JAX array, must have same shape as image.
    return_Fourier: whether to return the real or Fourier image.
  Returns:
    The resampled kimage.

  Note: This assumes both image and psf are sampled with same pixel scale!
  """

  if image.dtype in ['complex64', 'complex128']:
    kimage = image
  else:
    kimage = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(image)))  
  imkpsf = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(psf)))

  im_conv = kimage * imkpsf

  if return_Fourier:
    return im_conv
  else:
    return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(im_conv)).real)


