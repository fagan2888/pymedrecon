"""An wrapper class for the Discrete Fourier Transform.

This class is a wrapper class for the numpy fft operations with user-specified
image support parameters.
"""
import numpy as np


class MriDft(object):
    """A Discrete Fourier Transform object.

    Since the name of the function is "MriDft" rather than "Dft", this function
    takes the creative liberty of applying fftshifts and ifftshifts by default,
    i.e., it assumes that the inputs have the zero coordinate at the center of
    the array.

    Args:
        axes (int or array of ints, default: all): A list of axes over which to
            perform ffts and fftshifts.
        ifftshift (boolean, default: True): Boolean determining whether to
            perform ifftshift before fft/ifft.
        fftshift (boolean, default: True): Boolean determining whether to
            perform fftshift after fft/ifft.
        orthogonal (boolean, default: True): Boolean for unitary or non-
            unitary FFT.

    Examples:
        Initialization:

            >>> dftob = MriDft()

        Forward interpolation:

            >>> kdat = dftob.forw(image)

        Backward interpolation:

            >>> image = dftob.back(kdat)
    """

    def __init__(
            self, axes=None, ifftshift=True, fftshift=True, orthogonal=False):
        # load in parameters, compute defaults
        self.axes = axes
        self.ifftshift = ifftshift
        self.fftshift = fftshift
        if orthogonal:
            self.norm = "ortho"
        else:
            self.norm = None

    def forw(self, x):
        """Apply forward DFT.

        Args:
            x (array_like): The image to be DFTed.

        Returns:
            y (array_like): The frequency content of the image.
        """
        if self.ifftshift:
            x = np.fft.ifftshift(x, axes=self.axes)

        y = np.fft.fftn(x, axes=self.axes, norm=self.norm)

        if self.fftshift:
            y = np.fft.fftshift(y, axes=self.axes)

        return y

    def back(self, y):
        """Apply inverse DFT.

        Args:
            y (array_like): The frequency content to be iDFTed.

        Returns:
            x (array_like): The image.
        """
        if self.ifftshift:
            y = np.fft.ifftshift(y, axes=self.axes)

        x = np.fft.ifftn(y, axes=self.axes, norm=self.norm)
        if self.norm != "ortho":
            if self.axes is None:
                axlist = range(len(x.shape))
            else:
                axlist = self.axes
            normfact = 1
            for curax in axlist:
                normfact *= x.shape[curax]
            x = x * normfact

        if self.fftshift:
            x = np.fft.fftshift(x, axes=self.axes)

        return x

    def __repr__(self):
        out = '\nMriDft object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            out = out + '   {}: {}\n'.format(attr, value)

        return out


def main():
    """No arguments, runs a testing script."""
    print('running test script')

    from skimage import data
    import matplotlib.pyplot as plt

    x = np.double(data.camera()[::2, ::2])
    x_noise = x + \
        np.random.normal(size=x.shape) + 1j*np.random.normal(size=x.shape)

    dftob = MriDft()

    print(dftob)

    y = dftob.forw(x_noise)

    xest = dftob.back(y)

    plt.figure(1)
    plt.imshow(np.absolute(x))
    plt.gray()
    plt.title('Orignal image')

    plt.figure(2)
    plt.imshow(np.log10(np.absolute(y)))
    plt.gray()
    plt.title('Noisy x k-space')

    plt.figure(3)
    plt.imshow(np.log10(np.absolute(xest)))
    plt.gray()
    plt.title('Inverse DFT of k-space')
    plt.show()

    return xest


if __name__ == '__main__':
    main()
