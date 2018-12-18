"""Non-uniform FFTs based on Kaiser-Bessel interpolation.

The KbNufft class is an object that can compute the Non-uniform Fast Fourier
Transform. The process involves first applying a FFT to a d-dimensional
object, then interpolating to scattered point coordinates. Alternatively, this
object could also be used to interpolate from scattered point coordinates to
the grid and inverse FFT. The locations of the scattered points are denoted by
the k-space trajectory, which is a list of coordinates in digital frequency
space.

This package is implemented entirely in native Python. Although this makes it
platform-flexible, if more speed is desired, more optimized packages can be
found elsewhere.

Standard usage is as follows:
    Initialization:

        >>> kbnufftob = KbNufft(traj, im_size)

    Forward interpolation:

        >>> scat_dat = knufftob.forw(image)

    Backward interpolation:

        >>> image = knufftob.back(scat_dat)

Examine documentation of the KbNufft class for more details and arguments.
KbInterp can also be used separately for interpolation alone (i.e., no FFTs).

This implementation is based on the Matlab implmentation of Fessler,
available at https://web.eecs.umich.edu/~fessler/code/
"""
import itertools

import numpy as np
from scipy import special
from scipy.sparse import coo_matrix


class KbNufft(object):
    """Non-uniform FFT object via Kaiser-Bessel interpolation.

    This object computes the Non-uniform Fast Fourier Transform and its
    adjoint operation based on the user's k-space trajectory ('traj') and image
    dimensions. The traj array should be in radians, typically a numpy array of
    dimension (d, m), where m is the length of the trajectory and d is the
    number of dimensions. The trajectory will typically be scaled between -pi
    and pi.

    Args:
        traj (array_like): The k-space trajectory radians of size (d, m), where
            m is the length of the trajectory and d is the number of dimensions.
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate to.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        usetable (boolean, default=True): Use table interpolation instead of
            sparse matrix, saving memory.
        table_oversamp (int, default=2^10): Table oversampling factor.
        order (ind, default=0): Order of Kaiser-Bessel kernel. Not currently
            implemented.

    Examples:
        Initialization:

            >>> kbnufftob = KbNufft(traj, im_size)

        Forward interpolation:

            >>> scat_dat = knufftob.forw(image)

        Backward interpolation:

            >>> image = knufftob.back(scat_dat)
    """

    def __init__(
            self, traj, im_size, grid_size=None, numpoints=6, n_shift=None,
            usetable=True, table_oversamp=2**10, order=0):
        # load in parameters, compute defaults
        self.alpha = 2.34
        self.traj = traj
        self.im_size = im_size
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if numpoints is 6:
            self.numpoints = (6,) * len(self.grid_size)
        elif len(numpoints) is not len(self.grid_size):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.order = (0,)
        self.alpha = (2.34 * self.numpoints[0],)
        for i in range(1, len(self.numpoints)):
            self.alpha = self.alpha + (2.34 * self.numpoints[i],)
            self.order = self.order + (0,)
        self.usetable = False

        if usetable:
            self.usetable = usetable
            if table_oversamp == 2**10:
                self.table_oversamp = (table_oversamp,) * len(self.grid_size)
            else:
                self.table_oversamp = table_oversamp
        else:
            self.table_oversamp = None

        # build the interpolation object
        self.kb_interp_ob = KbInterp(
            traj=self.traj,
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=self.numpoints,
            n_shift=self.n_shift,
            usetable=self.usetable,
            table_oversamp=self.table_oversamp,
            order=self.order
        )

        # compute kb scaling coefficients
        num_coefs = np.array(range(self.im_size[0])) - (self.im_size[0]-1)/2
        self.scaling_coef = 1 / self._kaiser_bessel_ft(
            num_coefs/self.grid_size[0],
            self.numpoints[0],
            self.alpha[0],
            self.order[0],
            1
        )
        for i in range(1, traj.shape[0]):
            indlist = np.array(range(self.im_size[i])) - (self.im_size[i]-1)/2
            self.scaling_coef = np.expand_dims(self.scaling_coef, axis=-1)
            tmp = 1 / self._kaiser_bessel_ft(
                indlist/self.grid_size[i],
                self.numpoints[i],
                self.alpha[i],
                self.order[i],
                1
            )

            for _ in range(i):
                tmp = tmp[np.newaxis]

            self.scaling_coef = self.scaling_coef * tmp

    def _kaiser_bessel_ft(
            self, om, npts=None, alpha=None, order=None, d=None):
        """Computes FT of KB function for scaling in image domain."""
        if npts is None:
            npts = self.numpoints
        if alpha is None:
            alpha = self.alpha
        if order is None:
            order = self.order
        if d is None:
            d = 1

        z = np.sqrt((2*np.pi*(npts/2)*om)**2 - alpha**2 + 0j)
        nu = d/2 + order
        scaling_coef = (2*np.pi)**(d/2) * ((npts/2)**d) * (alpha**order) / \
            special.iv(order, alpha) * special.jv(nu, z) / (z**nu)
        scaling_coef = np.real(scaling_coef)

        return scaling_coef

    def forw(self, x, om=None):
        """Apply forward NUFFT.

        Args:
            x (array_like): The image to be NUFFTed.
            om (array_like, optional): A new set of frequency locations to
                compute.

        Returns:
            y (array_like): The frequency coefficients of x computed at
                off-grid locations.
        """
        x = x * self.scaling_coef
        y = self.kb_interp_ob.forw(np.fft.fftn(x, s=self.grid_size), om=om)

        return y

    def back(self, y, om=None):
        """Apply backward NUFFT.

        Args:
            y (array_like): The off-grid k-space signal.
            om (array_like, optional): A new set of frequency locations to
                interpolate from.

        Returns:
            x (array_like): The space-domain signal based on the off-grid
                frequency coefficients in y.
        """
        x = np.fft.ifftn(self.kb_interp_ob.back(y, om=om))
        x = x * np.prod(np.array(x.shape))
        startinds = tuple(np.array(self.im_size) * 0)
        x = x[tuple(map(slice, startinds, self.im_size))]
        x = x * np.conj(self.scaling_coef)

        return x

    def __repr__(self):
        out = '\nKbNufft NUFFT object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'scaling_coef' in attr:
                out = out + '   scaling_coef: {} {} array\n'.format(
                    self.scaling_coef.shape, self.scaling_coef.dtype)
            elif 'traj' in attr:
                out = out + '   traj: {} {} array\n'.format(
                    self.traj.shape, self.traj.dtype)
            elif 'kb_interp_ob' not in attr:
                out = out + '   {}: {}\n'.format(attr, value)

        out = out + '   kb_interp_ob: KbInterp Object\n'
        return out


class KbInterp(object):
    """Kaiser-Bessel interpolation between gridded data and off-grid locations.

    This object is meant to be used in conjuction with an FFT operation for
    computing the Non-uniform Fast Fourier Transform. It interpolates from
    gridded FFT data to off-grid points in the forward operation or from off-
    grid data to grid locations in the backward operation. The interpolation
    kernel is the Kaiser-Bessel function.

    The traj object should be in radians, typically a numpy array of dimension
    (d, m), where m is the length of the trajectory and d is the number of
    dimensions. The trajectory will typically be scaled between -pi and pi.

    Args:
        traj (array_like): The k-space trajectory radians of size (d, m),
            where m is the length of the trajectory and d is the number of
            dimensions.
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate to.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in
            each direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points
            to shift for fftshifts.
        usetable (boolean, default=True): Use table interpolation instead of
            sparse matrix.
        table_oversamp (int, default=2^10): Table oversampling factor.
        order (ind, default=0): Order of Kaiser-Bessel kernel. Not currently
            implemented.

    Examples:
        Initialization:

            >>> kbob = KbInterp(traj, im_size)

        Forward interpolation:

            >>> scat_dat = kbob.forw(grid_dat)

        Backward interpolation:

            >>> grid_dat = kbob.back(scat_dat)
    """

    def __init__(
            self, traj, im_size, grid_size=None, numpoints=6, n_shift=None,
            usetable=True, table_oversamp=2**10, order=0):
        self.alpha = 2.34
        self.traj = traj
        self.im_size = im_size
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if numpoints is 6:
            self.numpoints = (6,) * len(self.grid_size)
        elif len(numpoints) is not len(self.grid_size):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.order = (0,)
        self.alpha = (2.34 * self.numpoints[0],)
        for i in range(1, len(self.numpoints)):
            self.alpha = self.alpha + (2.34 * self.numpoints[i],)
            self.order = self.order + (0,)
        self.usetable = usetable

        if self.usetable:
            if table_oversamp == 2**10:
                self.table_oversamp = (table_oversamp,) * len(self.grid_size)
            else:
                self.table_oversamp = table_oversamp
            self.table = self._build_table()
        else:
            self.spmat = self._build_spmatrix()

    def _build_spmatrix(
            self, om=None, numpoints=None, im_size=None, grid_size=None,
            n_shift=None, order=None):
        """Builds a sparse matrix with the interpolation coefficients.

        This function is not meant to be used by the user in most cases; it is
        primarily a utility for building an interpolation table. By default,
        parameters are inferred from those passed to __init__ routine.
        """
        spmat = -1

        if om is None:
            om = self.traj
        if numpoints is None:
            numpoints = self.numpoints
        if im_size is None:
            im_size = self.im_size
        if grid_size is None:
            grid_size = self.grid_size
        if n_shift is None:
            n_shift = self.n_shift

        ndims = om.shape[0]
        klength = om.shape[1]

        # calculate interpolation coefficients using kb kernel
        def interp_coeff(om, npts, grdsz, alpha):
            gam = 2*np.pi / grdsz
            interp_dist = om / gam - np.floor(om / gam - npts/2)
            Jvec = np.reshape(np.array(range(1, npts+1)), (1, npts))
            kern_in = -1*Jvec + np.expand_dims(interp_dist, 1)

            cur_coeff = np.zeros(shape=kern_in.shape, dtype=complex)
            indices = abs(kern_in) < npts/2
            bess_arg = np.sqrt(1 - (kern_in[indices] / (npts/2))**2)
            denom = special.iv(0, alpha)
            cur_coeff[indices] = special.iv(0, alpha * bess_arg) / \
                denom
            cur_coeff = np.real(cur_coeff)
            return cur_coeff, kern_in

        full_coef = []
        kd = []
        for i in range(ndims):
            N = im_size[i]
            J = numpoints[i]
            K = grid_size[i]

            # get the interpolation coefficients
            coef, kern_in = interp_coeff(om[i, :], J, K, self.alpha[i])

            gam = 2*np.pi/K
            phase_scale = 1j * gam * (N-1)/2

            phase = np.exp(phase_scale * kern_in)
            full_coef.append(phase * coef)

            # nufft_offset
            koff = np.expand_dims(np.floor(om[i, :] / gam - J/2), 1)
            Jvec = np.reshape(np.array(range(1, J+1)), (1, J))
            kd.append(np.mod(Jvec + koff, K) + 1)

        for i in range(len(kd)):
            kd[i] = (kd[i] - 1) * np.prod(grid_size[i+1:])

        # build the sparse matrix
        kk = kd[0]
        spmat_coef = full_coef[0]
        for i in range(1, ndims):
            Jprod = np.prod(numpoints[:i+1])
            # block outer sum
            kk = np.reshape(
                np.expand_dims(kk, 1) + np.expand_dims(kd[i], 2),
                (klength, Jprod)
            )
            # block outer prod
            spmat_coef = np.reshape(
                np.expand_dims(spmat_coef, 1) *
                np.expand_dims(full_coef[i], 2),
                (klength, Jprod)
            )

        # build in fftshift
        phase = np.exp(1j * np.dot(np.transpose(om),
                                   np.expand_dims(n_shift, 1)))
        spmat_coef = np.conj(spmat_coef) * phase

        # get coordinates in sparse matrix
        trajind = np.expand_dims(np.array(range(klength)), 1)
        trajind = np.repeat(trajind, np.prod(numpoints), axis=1)

        # build the sparse matrix
        spmat = coo_matrix((
            spmat_coef.flatten(),
            (trajind.flatten(), kk.flatten())),
            shape=(klength, np.prod(grid_size))
        )

        return spmat

    def _build_table(self):
        """Builds a table for interpolation coefficient look-up.

        Parameters are inferred from __init__ routine.
        """
        ndims = self.traj.shape[0]
        table = []

        # build one table for each dimension
        for i in range(ndims):
            J = self.numpoints[i]
            L = self.table_oversamp[i]
            K = self.grid_size[i]
            N = self.im_size[i]

            # The following is a trick of Fessler.
            # It uses broadcasting semantics to quickly build the table.
            t1 = J/2-1 + np.array(range(L)) / L  # [L]
            om1 = t1 * 2*np.pi/K  # gam
            s1 = self._build_spmatrix(
                np.expand_dims(om1, 0),
                numpoints=(J,),
                im_size=(N,),
                grid_size=(K,),
                n_shift=(0,)
            )
            h = np.array(s1.getcol(J-1).todense())
            for col in range(J-2, -1, -1):
                h = np.concatenate(
                    (h, np.array(s1.getcol(col).todense())), axis=0)
            h = np.concatenate((h.flatten(), np.array([0])))

            table.append(h)

        return table

    def _table_interp(self, x, om=None):
        """Apply table interpolation.

        This function is flexible - one can pass in a new set of om k-space
        points and interpolate to locations not passed to original __init__
        routine.

        Args:
            x (array_like): The oversampled DFT of the signal.
            om (array_like, optional): A custom set of k-space points to
                interpolate to.
        """
        def run_interp(
                griddat, dims, table, numpoints, Jlist, L, tm, kdat):
            M = tm.shape[1]
            ndims = tm.shape[0]
            nJ = Jlist.shape[1]

            # center of tables
            centers = np.floor(numpoints*L/2).astype(np.int)
            # offset from k-space to first coef loc
            kofflist = 1 + np.floor(tm - np.expand_dims(numpoints, 1) / 2.0)

            # do a bit of type management - ints for faster index comps
            curgridind = np.zeros(tm.shape, dtype=np.int)
            curdistind = np.zeros(tm.shape, dtype=np.int)
            coef = np.ones((M,), dtype=np.complex)
            arr_ind = np.zeros((M,), dtype=np.int)
            dims = dims.astype(np.int)
            kofflist = kofflist.astype(np.int)
            Jlist = Jlist.astype(np.int)

            # loop over offsets and take advantage of numpy broadcasting
            for Jind in range(nJ):
                curgridind = kofflist + np.expand_dims(Jlist[:, Jind], 1)
                curdistind = np.rint(
                    (tm - curgridind) * np.expand_dims(L, 1)).astype(np.int)

                coef[:] = 1.0 + 0j
                arr_ind[:] = 0

                for d in range(ndims):
                    coef *= table[d, curdistind[d, :]+centers[d]]
                    arr_ind += np.mod(curgridind[d, :], dims[d]).flatten() * \
                        np.prod(dims[d+1:])

                # no danger of collisions for forward op
                kdat += coef * griddat[arr_ind]

            return kdat

        if om is None:
            om = self.traj

        ndims = om.shape[0]
        M = om.shape[1]

        # convert to normalized freq locs
        tm = np.zeros(shape=om.shape)
        Jgen = []
        for i in range(ndims):
            gam = 2*np.pi / self.grid_size[i]
            tm[i, :] = om[i, :] / gam
            Jgen.append(range(self.numpoints[i]))

        # build an iterator for going over all J values
        Jgen = list(itertools.product(*Jgen))

        # run the table interpolator
        y = run_interp(
            x.flatten(),
            np.array(x.shape),
            np.array(self.table),
            np.array(self.numpoints),
            np.transpose(np.array(Jgen)),
            np.array(self.table_oversamp),
            tm,
            np.zeros(shape=(M,), dtype=np.complex)
        )

        # phase for fftshift
        phase = np.exp(1j * np.dot(
            np.transpose(om),
            np.expand_dims(self.n_shift, 1)
        )
        )
        y = y*phase.flatten()

        return y

    def _table_interp_adj(self, y, om=None):
        """Apply table interpolation adjoint.

        This function is flexible - one can pass in a new set of om k-space
        points and interpolate from locations not passed to original __init__
        routine.

        Args:
            y (array_like): The off-grid k-space samples.
            om (array_like, optional): A custom set of k-space points to
                interpolate from.
        """
        def run_interp_adj(
                kdat, dims, table, numpoints, Jlist, L, tm, griddat):
            M = tm.shape[1]
            ndims = tm.shape[0]
            nJ = Jlist.shape[1]

            # center of tables
            centers = np.floor(numpoints*L/2).astype(np.int)
            # offset from k-space to first coef loc
            kofflist = 1 + np.floor(tm - np.expand_dims(numpoints, 1) / 2.0)

            # do a bit of type management - ints for faster index comps
            curgridind = np.zeros(tm.shape, dtype=np.int)
            curdistind = np.zeros(tm.shape, dtype=np.int)
            coef = np.ones((M,), dtype=np.complex)
            arr_ind = np.zeros((M,), dtype=np.int)
            dims = dims.astype(np.int)
            kofflist = kofflist.astype(np.int)
            Jlist = Jlist.astype(np.int)

            # loop over offsets and take advantage of numpy broadcasting
            for Jind in range(nJ):
                curgridind = kofflist + np.expand_dims(Jlist[:, Jind], 1)
                curdistind = np.rint(
                    (tm - curgridind) * np.expand_dims(L, 1)).astype(np.int)

                coef[:] = 1.0 + 0j
                arr_ind[:] = 0

                for d in range(ndims):
                    coef *= np.conj(table[d, curdistind[d, :]+centers[d]])
                    arr_ind += np.mod(curgridind[d, :], dims[d]).flatten() * \
                        np.prod(dims[d+1:])

                # avoid write collisions by using bincount
                tmp = np.bincount(arr_ind, np.real(coef*kdat)) + \
                    1j*np.bincount(arr_ind, np.imag(coef*kdat))
                griddat[:len(tmp)] += tmp

            return griddat

        if om is None:
            om = self.traj

        ndims = om.shape[0]

        # phase for fftshift
        phase = np.exp(1j * np.dot(
            np.transpose(om),
            np.expand_dims(self.n_shift, 1)
        )
        )
        y = y * np.conj(phase.flatten())

        # convert to normalized freq locs
        tm = np.zeros(shape=om.shape)
        Jgen = []
        for i in range(ndims):
            gam = 2*np.pi / self.grid_size[i]
            tm[i, :] = om[i, :] / gam
            Jgen.append(range(self.numpoints[i]))

        # build an iterator for going over all J values
        Jgen = list(itertools.product(*Jgen))

        # run the table adjoint
        x = run_interp_adj(
            y,
            np.array(self.grid_size),
            np.array(self.table),
            np.array(self.numpoints),
            np.transpose(np.array(Jgen)),
            np.array(self.table_oversamp),
            tm,
            np.zeros(shape=self.grid_size, dtype=np.complex).flatten()
        )

        return x

    def forw(self, x, om=None):
        """Interpolate from gridded data to scattered data.

        Args:
            x (array_like): The oversampled DFT of the signal.
            om (array_like, optional): A new set of omega coordinates to
                interpolate to. Only available with table option.
        Returns:
            y (array_like): x computed at off-grid locations in om.
        """
        if self.usetable:
            y = self._table_interp(np.reshape(x, self.grid_size), om=om)
        elif om is not None:
            print('custom trajectory only available with table option!')
            y = -1
        else:
            y = self.spmat.dot(x.flatten())
        return y

    def back(self, y, om=None):
        """Interpolate from scattered data to gridded data.

        Args:
            y (array_like): The off-grid k-space signal.
            om (array_like, optional): A new set of omega coordinates to
                interpolate from. Only available with table option.
        Returns:
            x (array_like): y interpolated from off-grid locations in om
                to on-grid locations.
        """
        if self.usetable:
            x = np.reshape(self._table_interp_adj(y, om=om), self.grid_size)
        elif om is not None:
            print('new trajectory only available with table option!')
            x = -1
        else:
            x = np.reshape(self.spmat.getH().dot(y.flatten()), self.grid_size)
        return x

    def __repr__(self):
        tablecheck = False
        out = '\nKbInterp interpolation object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'table' in attr:
                if not tablecheck:
                    out = out + '   table: {} arrays, lengths: {}\n'.format(
                        len(self.table), self.table_oversamp)
                    tablecheck = True
            elif 'traj' in attr:
                out = out + '   traj: {} {} array\n'.format(
                    self.traj.shape, self.traj.dtype)
            else:
                out = out + '   {}: {}\n'.format(attr, value)
        return out


def main():
    """No arguments, runs a testing script."""
    print('running test script')
    writemat = False
    import time
    if writemat:
        import scipy.io as sio
    from skimage import data
    import matplotlib.pyplot as plt

    x = np.double(data.camera()[::2, ::2])

    spokelength = 256
    nspokes = 405

    ga = np.deg2rad(-111.25)
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga)*kx[:, i-1] - np.sin(ga)*ky[:, i-1]
        ky[:, i] = np.sin(ga)*kx[:, i-1] + np.cos(ga)*ky[:, i-1]

    # plt.plot(kx, ky)
    # plt.title('kspace trajectory')
    # plt.show()

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    traj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    im_size = tuple(x.shape)

    tableob = KbNufft(traj, im_size)
    print(tableob)
    matob = KbNufft(traj, im_size, usetable=False)

    startmat = time.time()
    y_mat = matob.forw(x)
    recon_mat = matob.back(y_mat)
    endmat = time.time()

    starttable = time.time()
    y_table = tableob.forw(x)
    endforw = time.time()
    print('table forw: {:.5f}'.format(endforw-starttable))
    recon_table = tableob.back(y_table)
    endtable = time.time()

    print('mat interp time: {:.5f}, table interp time: {:.5f}'.format(
        endmat-startmat, endtable-starttable))
    print('normdiff kspace: {}'.format(
        np.linalg.norm(y_mat-y_table)/np.linalg.norm(y_table)))
    print('normdiff recon: {}'.format(np.linalg.norm(
        recon_mat-recon_table)/np.linalg.norm(recon_table)))

    plt.figure(2)
    plt.imshow(np.absolute(recon_mat))
    plt.gray()
    plt.title('sparse matrix reconstruction')

    plt.figure(3)
    plt.imshow(np.absolute(recon_table))
    plt.gray()
    plt.title('table reconstruction')
    plt.show()

    if writemat:
        sio.savemat('nufft_test.mat', {
            'recon_mat': recon_mat,
            'recon_table': recon_table,
            'y_mat': y_mat.flatten(),
            'y_table': y_table.flatten(),
            'x': x,
            'ktraj': traj,
            'table': tableob.kb_interp_ob.table
        })


if __name__ == '__main__':
    main()
