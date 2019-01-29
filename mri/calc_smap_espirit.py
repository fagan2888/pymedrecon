"""Calculation of sensitivity maps via ESPIRIT algorithm."""
import itertools

import numpy as np


def calc_smap_espirit(
        data, ncalib, ksize=6, eigthresh1=0.02, eigenthresh2=0.95):
    """ Calculate sensitivity maps via ESPIRIT. Based on the implementation of
        Uecker/Lustig.

        Args:
            data (array_like): k-space data for sensitivity map estimation,
                first dimension is coil dimension
            ksize (int or tuple of ints): size of calibration kernel
            ncalib (int or tuple of ints): size of calibration region (second dimension
                if int)
            eigthresh1 (double): first eigenvalue threshold
            eigthresh2 (double): second eigenvalue threshold

        Returns:
            coil_array (array_like): An array of dimensions (ncoils (N)),
                specifying spatially-varying sensitivity maps for each coil.
    """
    if len(np.atleast_1d(ncalib).shape) == 1:
        center = data.shape[1] // 2
        calib_region = data[:, center - ncalib //
                            2: center-ncalib//2+ncalib, ...]
    else:
        calib_region = data
        for i, dimsize in enumerate(ncalib.shape):
            center = calib_region.shape[i+1] // 2
            calib_region = np.take(
                calib_region,
                list(range(center-dimsize//2, center-dimsize//2+dimsize)),
                axis=i+1
            )

    # take into account coildim
    if len(np.atleast_1d(ksize).shape) != (len(data.shape) - 1):
        ksize = tuple(
            (np.array(ksize) * np.ones((len(data.shape)-1,))).astype(np.int))

    # kernels.shape = (nblocks, ncoils, (ksize))
    kernels = extract_kernel(calib_region, ksize, eigthresh1)

    smap = extract_maps(kernels, data.shape[1:], eigenthresh2)

    return smap


def extract_blocks(calib_region, ksize):
    """Extract blocks from the calibration region.

    Args:
        calib_region (array_like): Extracted calibration data.
        ksize (tuple): Size of kernels for ESPIRiT calculation.
    Returns:
        blocks (array_like): Blocks extracted from calibration data of size
            (nblocks, ncoil, (imdim))
    """
    block_loop_shifts = []
    block_loop_offsets = []
    for i, dim in enumerate(ksize):
        block_loop_shifts.append(range(dim))
        block_loop_offsets.append(range(0, calib_region.shape[i+1], dim))
    block_loop_shifts = list(itertools.product(*block_loop_shifts))
    block_loop_offsets = np.array(
        list(itertools.product(*block_loop_offsets))).T

    blocks = []
    for shift in block_loop_shifts:
        current_offsets = np.expand_dims(
            np.array(shift), 1) + block_loop_offsets
        current_extract_inds = np.expand_dims(current_offsets, -1) + \
            np.expand_dims(np.array(block_loop_shifts).T, 1)
        for i in range(len(ksize)):
            mask = current_extract_inds[i, :, -1] < calib_region.shape[i+1]
            current_extract_inds = current_extract_inds[:, mask, ...]
        current_extract_inds = tuple(np.reshape(
            current_extract_inds, (len(ksize), -1)))

        tmp = []
        for coil in calib_region:  # loop over sensitivity coils
            tmp.append(
                np.reshape(
                    coil[current_extract_inds],
                    (-1, np.prod(ksize))
                )
            )

        # append (nblocks/nshifts, ncoil, prod(ksize)) array
        blocks.append(np.stack(tmp, 1))

    return np.concatenate(blocks, 0)  # (nblocks, ncoil, prod(ksize))


def extract_kernel(calib_region, ksize, eigthresh1):
    """Calculate the kernels via SVD.

    Args:
        calib_region (array_like): Extracted calibration data.
        ksize (tuple): Size of kernels for ESPIRiT calculation.
        eigthresh1 (double): Threshold for removing eigenvalues below
            maxeig*eigthresh1.
    Returns:
        kernels (array_like): Kernels extracted from calibration data of size
            (nkernels, ncoil, (ksize))
    """
    blocks = extract_blocks(calib_region, ksize)

    # calculate svd and get kernels
    blocks = np.reshape(blocks, (blocks.shape[0], -1))
    _, singular_values, kernels = np.linalg.svd(blocks, full_matrices=False)
    kernels = np.reshape(
        kernels,
        (-1, calib_region.shape[0]) + ksize
    )

    # threshold the kernels based on the first eigthresh
    kernel_cutoff = np.searchsorted(
        -1*singular_values,
        -1*eigthresh1*np.max(singular_values)
    )
    kernels = kernels[:kernel_cutoff, ...]

    return kernels  # (nkernels, ncoil, (ksize))


def extract_maps(kernels, imsize, eigenthresh2=0.97):
    """Calculate the kernel via SVD for each pixel in image.

    Args:
        kernels (array_like): Extracted ESPIRiT kernels.
        imsize (tuple): Size of image for sensitivities.
        eigthresh2 (double): Threshold for removing eigenvalues below
            eigthresh2 (should be close to 1).
    Returns:
        smap (array_like): ESPIRiT sensitivity maps (ncoil, (imdim))
    """
    padsizes = [(0, 0), (0, 0)]
    ksize = []
    for i, dim in enumerate(imsize):
        ksize.append(kernels.shape[i+2])
        padsizes.append(((dim - ksize[i])//2, (dim - ksize[i] + 1)//2))
    padsizes = tuple(padsizes)

    # convert to image domain
    fft_axes = tuple(2 + np.array(range(kernels.ndim-2)))
    kernels = np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(
                np.pad(kernels, padsizes, mode='constant', constant_values=0),
                axes=fft_axes
            ),
            axes=fft_axes
        ),
        axes=fft_axes
    ) * np.prod(imsize) / np.sqrt(np.prod(ksize))

    # ((imsize), ncoils, num_kernels)
    kernels = np.transpose(kernels)

    eigenvects, eigenvals, _ = np.linalg.svd(kernels, full_matrices=False)

    # normalize phase by setting first coil phase to 0
    eigenvects = eigenvects * \
        np.exp(-1j * np.angle(eigenvects[..., 0, np.newaxis, :]))

    # threshold eigenvalues, compute smaps
    eigenvals[eigenvals < eigenthresh2] = 0
    smap = np.expand_dims(eigenvals[..., 0], -1) * eigenvects[..., 0]

    smap = np.transpose(smap)

    return smap


def main():
    """No arguments, runs a testing script."""
    print('running test script')
    import sys
    import os
    import scipy.io as sio
    import matplotlib.pyplot as plt
    from skimage import data
    sys.path.append('./')
    from mri.systems.mrisys import Mri
    from mri.mrisensesim import mrisensesim

    x = np.double(data.camera()[::2, ::2]) + 0j

    smap = np.array(mrisensesim(size=x.shape, ncoils=15, coil_width=20))

    mriob = Mri(np.array(smap), multiprocessing=False)

    data = mriob.forw(x)

    smap_est = calc_smap_espirit(data, 24)

    fig = plt.figure(figsize=(2, 8))
    plt.gray()
    for i in range(2*4):
        fig.add_subplot(2, 4, i+1)
        plt.imshow(np.absolute(smap[i, ...]))
    plt.title('True sensitivities')

    fig = plt.figure(figsize=(2, 8))
    plt.gray()
    for i in range(2*4):
        fig.add_subplot(2, 4, i+1)
        plt.imshow(np.absolute(smap_est[i, ...]))
    plt.title('Estimated sensitivities')

    plt.show()


if __name__ == '__main__':
    main()
