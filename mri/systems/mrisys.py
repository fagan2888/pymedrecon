"""An MRI linear system model.

This function creates an object that computes the linear transformation from
image to k-space to approximate the physics of an MRI scanner. The Mri class is
minimalist in that it only has two components: sensitivity maps and an encoding
operator. By default, the encoding operator is an ND FFT operation - more
elaborate encoding schemes can be passed in if desired. See the Mri class for
details.
"""
from multiprocessing import Pool

import numpy as np

from mridft import MriDft


class Mri(object):
    """An MRI linear system model.

    The model is structured for a multi-coil MR system. Specification of smap
    is required. If a single-coil system is desired, put in a ones array of
    size [nc, (N)] for the smap, or, alternatively, use the desired encoding
    object directly.

    Args:
        smap (array_like): A complex array of dimension [nc, (N)], where (N)
            can be length-2 (for 2D), length-3 (for 3D), or more, and nc
            is the number of channels.
        encodeob (class, default: Cartesian FFT): An object for encoding each
            channel into k-space after applying sensitivity maps.
        multiprocessing (boolean, default: True): Use multiprocessing to
            parallelize encoding operator over coils.

    Examples:
        Initialization:

            >>> mriob = Mri(smap)

        Forward interpolation:

            >>> multicoil_dat = mriob.forw(image)

        Backward interpolation:

            >>> image = mriob.back(multicoil_dat)
    """

    def __init__(self, smap, encodeob=None, multiprocessing=True):
        # load in parameters, compute defaults
        self.smap = smap
        self.numcoils = smap.shape[0]
        self.imdims = smap.shape[1:]
        if encodeob is None:
            self.fftop = True
            self.encodeob = MriDft(axes=list(range(1, len(smap.shape))))
        else:
            self.fftop = False
            self.encodeob = encodeob
        self.multiprocessing = multiprocessing

    def forw(self, x):
        """Apply forward MRI encoding.

        Args:
            x (array_like): The image to be encoded.

        Returns:
            y (array_like): k-space data for all coils. Coil dimension will be
                the first dimension of the array.
        """
        x = np.expand_dims(np.reshape(x, self.imdims), 0) * self.smap

        if self.fftop:
            y = self.encodeob.forw(x)
        elif self.multiprocessing:
            with Pool() as pool:
                y = np.array(pool.map(self.encodeob.forw, list(x)))
        else:
            y = []
            for i in range(self.numcoils):
                y.append(self.encodeob.forw(x[i, ...]))
            y = np.array(y)

        return y

    def back(self, y):
        """Apply adjoint MRI encoding.

        Args:
            y (array_like): The encoded data for all coils. Coil dimension
                should be the first dimension of the array.

        Returns:
            x (array_like): Image after inverse encoding and conjugate coil
                sum.
        """
        if self.fftop:
            x = self.encodeob.back(y)
        elif self.multiprocessing:
            with Pool() as pool:
                x = np.array(pool.map(self.encodeob.back, list(y)))
        else:
            x = []
            for i in range(self.numcoils):
                x.append(self.encodeob.back(y[i, ...]))
            x = np.array(x)

        x = np.sum(x * np.conj(self.smap), axis=0)

        return x

    def __repr__(self):
        out = '\nMri object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'smap' in attr:
                out = out + '   smap: {} {} array\n'.format(
                    self.smap.shape, self.smap.dtype)
            else:
                out = out + '   {}: {}\n'.format(attr, value)

        out = out + '   encodeob: Encoding Object\n'
        return out


def main():
    """No arguments, runs a testing script."""
    import sys
    from skimage import data
    import matplotlib.pyplot as plt
    sys.path.append('./')
    from mri import mrisensesim
    from kbnufft import KbNufft

    print('running test script')

    x = np.double(data.camera()[::2, ::2]) + 0j

    print('imsize: {}'.format(x.shape))

    spokelength = 256
    nspokes = 405
    ga = np.deg2rad(-111.25)
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga)*kx[:, i-1] - np.sin(ga)*ky[:, i-1]
        ky[:, i] = np.sin(ga)*kx[:, i-1] + np.cos(ga)*ky[:, i-1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)
    traj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    nufftob = KbNufft(traj, x.shape, usetable=False)

    smap = mrisensesim(size=x.shape, coil_width=20)

    mriob = Mri(np.array(smap), encodeob=nufftob, multiprocessing=False)

    y = mriob.forw(x)

    x = mriob.back(y)

    smap = np.reshape(np.array(smap), (8*x.shape[1], x.shape[0]))
    plt.figure(1)
    plt.gray()
    plt.imshow(np.absolute(smap))
    plt.title('sensitivity maps')

    plt.figure(2)
    plt.imshow(np.absolute(x))
    plt.gray()

    plt.show()

    return x


if __name__ == '__main__':
    main()
