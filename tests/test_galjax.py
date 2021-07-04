import galsim as gs
import quarks2cosmos.galjax as gj
from numpy.testing import assert_allclose

# Some parameters used for testing shearing transforms
gal_flux = 1.e5      # counts
gal_r0 = 0.1         # arcsec
e1 = 0.1             #
e2 = 0.5             #
pixel_scale = 0.03   # arcsec / pixel
target_scale = 0.168 # arcsec / pixel

def test_deconv_and_resample():
  gal = gs.Exponential(flux=gal_flux, scale_radius=gal_r0)
  gal = gal.shear(g1=e1,g2=e2)
  cosmos_psf = gs.Gaussian(fwhm=0.2)
  cosmos_gal = gs.Convolve(gal, cosmos_psf)

  hsc_psf = gs.Gaussian(fwhm=0.7)
  hsc_gal = gs.Convolve(gal, hsc_psf)

  # Generates the stamps
  cosmos_stamp = cosmos_gal.drawImage(nx=91, ny=91, scale=pixel_scale).array
  cosmos_psf_stamp = cosmos_psf.drawImage(nx=91, ny=91, scale=pixel_scale).array
  hsc_stamp = hsc_gal.drawImage(nx=33, ny=33, scale=target_scale).array
  hsc_psf_stamp = hsc_psf.drawImage(nx=33, ny=33, scale=target_scale).array

  # Uses GalSim to get the target image
  im_cosmos = gs.InterpolatedImage(gs.Image(cosmos_stamp, scale=pixel_scale))
  im_cosmos_psf = gs.InterpolatedImage(gs.Image(cosmos_psf_stamp, scale=pixel_scale))
  im_hsc_psf = gs.InterpolatedImage(gs.Image(hsc_psf_stamp, scale=target_scale))
  target_obs_deconv = gs.Convolve(im_cosmos, gs.Deconvolve(im_cosmos_psf)) 
  target_obs_reconv = gs.Convolve(target_obs_deconv, im_hsc_psf)
  target_obs = target_obs_reconv.drawImage(nx=33, ny=33, scale=target_scale, method='no_pixel').array

  # Same thing with galjax
  deconv_image = gj.deconvolve(cosmos_stamp, cosmos_psf_stamp)
  resampled_deconv_image = gj.kresample(deconv_image, pixel_scale,
                                        target_scale, 33)
  forward_image = gj.convolve(resampled_deconv_image, hsc_psf_stamp)

  assert_allclose(forward_image, target_obs, atol=deconv_image.max()*2e-3)

