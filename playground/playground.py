import healpy as hp
from healpy.newvisufunc import projview, newprojplot
import numpy as np
import torch
import matplotlib.pyplot as plt

nside = 2
npix = hp.nside2npix(nside)
#directions = hp.pix2vec(nside, np.arange(npix))
#directions = torch.tensor(directions).T  # Shape: (npix, 3)

zenith, azimuth = hp.pix2ang(nside,np.arange(npix))

m = np.arange(npix)

plt.figure()
# projview(
#     m,
#     graticule=True,
#     graticule_labels=True,
#     unit="likelihood ratio",
#     xlabel="azimuth",
#     ylabel="zenith",
#     cb_orientation="horizontal",
#     projection_type="lambert",
# )


newprojplot(phi=np.pi/2, theta=-np.pi/3, marker="x", color="r", markersize=10, label= 'test')

plt.legend()

#plt.xlim(0,2)

# plt.savefig('test.png')
print(len(zenith))
print(len(azimuth))


# azimuth, zenith = hp.vec2ang(directions.T.numpy())

#print(azimuth)
#print(zenith)