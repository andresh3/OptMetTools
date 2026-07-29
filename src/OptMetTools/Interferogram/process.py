
# Note: Process functions accept
# NumPy arrays, but use torch under
# the hood for increased processing speed.
import numpy as np
import torch as tc

class Process:
    """
    Interferogram.Process
    Define several algorithms for processing interferograms.

    Provides vectorized implementation, allowing for processing
    of many unrelated interferogram stacks.


    References:
        1. https://doi.org/10.1364/OE.27.037634
        2. https://doi.org/10.1016/j.optlaseng.2021.106768
        3. https://doi.org/10.1088/2040-8978/17/2/025704 
        4. http://doi.org/10.1080/09500349514550341
        5. https://doi.org/10.1364/OL.29.001671
        6. https://doi.org/10.1016/j.optlaseng.2005.11.003
    """

    MAX_ITER = 250
    @staticmethod
    def Window(I,N=5):
        """
        Returns the interferogram windowed into seperate groups, I1, I2, ..., IN

        Parameters:
            I : np.array(..., M)
                Interfergram where the last parameter is the scan index

        Returns:
            *Ik, k=1,2,...,N np.array(...,M-N+1)
                Shifted intensity maps.
        """
        arr = []
        for i in range(N-1):
            arr.append(I[...,i:1-N+i])
        arr.append(I[...,N-1:])
        return tuple(arr)

    @staticmethod
    def AIA(I, delta, epsilon, img_mask=None):
        """
        Perform the iterative AIA algorithm (for PSI) to uncover the phase and
        phase-steps.

        Parameters:
            I : np.array(T,Nx,Ny,M)
                T unrelated stacks of M interferograms
            delta : np.array(T,M)
                T unrelated sets of initial guesses for delta.
            epsilon : float
                Convergence parameter representing the maximum incremental
                change between delta values.
            img_mask : np.array(T,Nx,Ny)
                Defines a mask of valid pixels, so as to allow differently sized images to be
                processed together. Intensity array I, should be zero-padded where pixels
                are invalid. Value of None implies all pixels are valid. Default None.

        Returns:
            phi : np.array(T,Nx,Ny)
                Final calculated phase map.
            delta : np.array(T,M)
                Final phase-shift mapping.
            n_iter : np.array(T)
                Number of iterations that each batch/test (t in 1,2,...,T) took.

        Citations:
            [5]
            [6]
        """
        # Too lazy to properly remove these (they were "advancements" to the AIA
        # algorithm suggested in another paper). The original paper acknowledged
        # these terms and actually chose not to use them. The reason, is they suck
        # and don't help.
        use_alpha=False
        use_beta=False
        # Cast from numpy to torch:
        I = tc.from_numpy(I).to(tc.float32)
        delta = tc.from_numpy(delta).to(tc.float32)
        if img_mask is not None:
            img_mask = tc.from_numpy(img_mask).to(tc.bool)

        T, Nx, Ny, M = I.shape
        
        delta_last = tc.zeros(*delta.shape)
        n_iter     = tc.zeros(delta.shape[0])

        phi        = tc.zeros(T, Nx, Ny)

        # define mask to stop processing
        # batches that have converged:
        conv_mask  = tc.ones(T, dtype=bool)

        # Make an initial assumptio about the frame-contrast:
        alpha = tc.ones_like(delta)
        beta  = tc.ones_like(delta)

        while(n_iter.max() < Process.MAX_ITER):

            im_mask = None
            if img_mask is not None:
                im_mask = img_mask[conv_mask]

            phi[conv_mask]   = Process._aia_est_phase(I[conv_mask], delta[conv_mask], alpha[conv_mask], beta[conv_mask])
            delta[conv_mask], alpha[conv_mask], beta[conv_mask] = Process._aia_est_phase_step(I[conv_mask], phi[conv_mask],im_mask)
            n_iter[conv_mask] += 1

            # update user options:
            if not use_alpha:
                alpha = tc.ones_like(delta)
            if not use_beta:
                beta  = tc.ones_like(delta)

            # Check for convergence and update mask:
            conv_mask = tc.any(
                    tc.abs((delta - delta[:,0][:,None]) - (delta_last - delta_last[:,0][:,None])) > epsilon,
                    dim=1)

            # Convergence has been reached for all trials:
            if tc.sum(conv_mask) == 0: 
                break

            # Update delta_last
            delta_last = delta.detach().clone()

        # Set the first delta value as the "reference"
        delta = delta - delta[:,0][:, None]
        # Rewrap from (-pi, pi):
        delta = ((delta + tc.pi) % (2*tc.pi)) - tc.pi 
        # Force knowledge of ascending step-direction:
        delta[delta[:, 1] < 0] *= -1
        
        # Recalculate phi with updated delta map:
        if not use_alpha:
            alpha = tc.ones_like(delta)
        if not use_beta:
            beta  = tc.ones_like(delta)
        phi = Process._aia_est_phase(I, delta, alpha, beta)

        return (phi.numpy(), delta.numpy(), n_iter.numpy())

    @staticmethod
    def _aia_est_phase(I, delta, alpha, beta):
        """
        internal method for phase estimation step of aia algorithm

        Parameters:
            I : tc.tensor(T,Nx,Ny,M)
            delta : tc.tensor(T,M)
            alpha : tc.tensor(T,N)
            beta : tc.tensor(T,N)


        Citations:
            [1]
        """
        T, Nx, Ny, M = I.shape

        cos_delta = tc.cos(delta)
        sin_delta = tc.sin(delta)

        M_stack           = tc.sum(alpha**2,axis=1)
        sum_cos_delta     = tc.sum(alpha*beta*cos_delta,axis=1)
        sum_sin_delta     = tc.sum(alpha*beta*sin_delta,axis=1)
        sum_sin_cos       = tc.sum((beta**2)*sin_delta*cos_delta, axis=1)
        sum_cos_sqr       = tc.sum((beta**2)*cos_delta*cos_delta, axis=1)
        sum_sin_sqr       = tc.sum((beta**2)*sin_delta*sin_delta, axis=1)

        # Shape: (T,3,3)
        Ap = tc.stack([
            tc.stack([M_stack,       sum_cos_delta, sum_sin_delta]),
            tc.stack([sum_cos_delta, sum_cos_sqr,   sum_sin_cos]),
            tc.stack([sum_sin_delta, sum_sin_cos,   sum_sin_sqr])
        ]).movedim(-1,0)

        # Invert T batches of the 3x3 Ap matrix:
        Ap = tc.linalg.inv(Ap)

        # Shapes: (T,Nx,Ny):
        I_sum = tc.sum(I, dim=(3))
        I_sum_cos = tc.sum(I*cos_delta[:,None,None,:], dim=(3))
        I_sum_sin = tc.sum(I*sin_delta[:,None,None,:], dim=(3))

        # Shape: (T, 3, Nx, Ny)
        B = tc.stack([
            I_sum,
            I_sum_cos,
            I_sum_sin
        ]).movedim(1,0)

        # Perform the matrix multiplication
        # for each matrix Ap[t,:,:] @ B[t,:,x,y] [(3x3) X (3x1) - > (3x1)]
        # Shape: (T, 3, Nx, Ny):
        X = tc.einsum('tij,tjxy->tixy',Ap,B) 

        a, b, c = X[:,0], X[:,1], X[:,2]

        phi = tc.atan2(-c,b)
        
        return phi


    @staticmethod
    def _aia_est_phase_step(I, phi, img_mask):
        """
        internal method for phase step estimation step of aia algorithm

        Parameters:
            I : tc.tensor(T,Nx,Ny,M)
            phi : tc.tensor(T,Nx,Ny)
            img_mask: tc.tensor(T,Nx,Ny) or None


        Citations:
            [1]
        """
        T, Nx, Ny, M = I.shape

        if img_mask is None:
            Npx_stack = tc.full((T,),Nx*Ny)
        else:
            Npx_stack = img_mask.sum(dim=(1,2)).to(tc.float32)

        cos_phi    = tc.cos(phi)
        sin_phi    = tc.sin(phi)

        if img_mask is not None:
            cos_phi[~img_mask] = 0
            sin_phi[~img_mask] = 0

        sum_cos_phi     = tc.sum(cos_phi,dim=(1,2))
        sum_sin_phi     = tc.sum(sin_phi,dim=(1,2))
        sum_sin_cos       = tc.sum(sin_phi*cos_phi, dim=(1,2))
        sum_cos_sqr       = tc.sum(cos_phi*cos_phi, dim=(1,2))
        sum_sin_sqr       = tc.sum(sin_phi*sin_phi, dim=(1,2))

        # Shape: (T,3,3)
        Ap = tc.stack([
            tc.stack([Npx_stack,   sum_cos_phi,   sum_sin_phi]),
            tc.stack([sum_cos_phi, sum_cos_sqr,   sum_sin_cos]),
            tc.stack([sum_sin_phi, sum_sin_cos,   sum_sin_sqr])
        ]).movedim(-1,0)


        # Invert T batches of the 3x3 Ap matrix:
        Ap = tc.linalg.inv(Ap)

        # Shapes: (T,M):
        I_sum = tc.sum(I, dim=(1,2))
        I_sum_cos = tc.sum(I*cos_phi[:,:,:,None], dim=(1,2))
        I_sum_sin = tc.sum(I*sin_phi[:,:,:,None], dim=(1,2))

        # Shape: (T, 3, M)
        B = tc.stack([
            I_sum,
            I_sum_cos,
            I_sum_sin
        ]).movedim(1,0)

        # Shape: (T, 3, M):
        X = tc.einsum('tij,tjm->tim',Ap,B) 

        a, b, c = X[:,0], X[:,1], X[:,2]

        # Shape: (T,M)
        delta = tc.atan2(-c,b)
        alpha    = a
        beta     = tc.sqrt(b**2 + c**2)
        
        return delta, alpha, beta


    def FDA(I, step_size, lambda0, n=1, ng=1, n_wavenumbers=3):
        """
        Performs Frequency Domain Analysis on the provided interferogram stack, I
        with step_size delta between adjacent scans. 

        Parameters:
            I : np.array(Nx,Ny,M)
                Stack of M interferograms.
            step_size : float
                The step size between adjacent scans, in microns.
            lambda0 : float
                Central wavelength of the light source, in microns.
            n : float (default 1.0)
                index of refraction of sampled material
            ng : float (default = 1.0)
                group velocity index
            n_wavenumbers : int (default = 3)
               The number of seperate wavenumbers that are used to
               evaluate the initial height estimate G0.

        Returns:
            surface : np.array(Nx,Ny)
                Phase-corrected surface height.
            z_est : np.array(Nx,Ny)
                Initial surface height estimate, based off
                of the group-velocity index G0.

        Citations:
            [4]
        """
        nx,ny,nz = I.shape
        npx = nx*ny
        I = I.copy() # don't corrupt users data!

        k0 = (4 * np.pi) / lambda0
        scanning_range = step_size * nz
        delta_k = np.pi / (2*scanning_range)

        # Zero mean the data
        I -= np.mean(I, axis=-1, keepdims=True)

        # Perform DFT for desired frequencies:
        freqs = k0 + delta_k * (np.arange(n_wavenumbers) - ((n_wavenumbers - 1) // 2))
        steps = np.arange(nz) * step_size

        I_f = np.sum(
            I[:,:,:,None] * np.exp(1j * freqs[None,:] * (steps)[:,None])[None,None,:,:],
            axis=-2
        )
        phi_k = np.unwrap(np.angle(I_f), axis=-1)

        # Coefficient matrix (nk, 2)
        X = np.ones((n_wavenumbers,2))
        X[:,0] = freqs

        # Response matrix (observed values for each input) (2, nk)
        Y = np.moveaxis(phi_k.reshape(npx,n_wavenumbers), -1, 0)

        # Find the coefficients (slope + intercept):
        B, _, _, _ = np.linalg.lstsq(X,Y)
        G0 = B[0,:].reshape(nx,ny)
        
        # Unwrap the phase map given by the FFT about k0:
        phase_unwrap = np.angle(I_f[..., n_wavenumbers//2]) 
        phase_est = G0 * k0

        i = 0 
        while np.any(np.abs(phase_unwrap - phase_est) > (np.pi)):
            phase_unwrap[(phase_unwrap - phase_est) > (np.pi)] -= (2*np.pi)
            phase_unwrap[(phase_unwrap - phase_est) < (-np.pi)] += (2*np.pi)

            i += 1
            if( i > 100 ):
                raise Exception("An error occured unwrapping...")

        phase_unwrap /= k0

        return (phase_unwrap, G0)


