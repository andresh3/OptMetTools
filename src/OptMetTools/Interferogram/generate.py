import numpy as np

class Generate:
    """
    Interferogram.Generate module has methods useful for generating interferograms
    """

    @staticmethod
    def WLI(Z,scanning_steps,lambda0, Lc, A=0.5, B=0.5, noise=0, phase_offset=0):
        """
        Generates a white-light interferogram stack. Input height is in
        objective units (i.e., Micron).

        Parameters:
            Z : np.array(Nx,Ny)
                Input surface.
            scanning_steps : np.array(Nz)
                Array representing the height (or phase) of each scan.
            lambda0 : float
                Central wavelength of light.
            Lc : float
                Coherence length.
            A : float
                Background illumnination constant. Default 0.5.
            B : float
                Modulation contrast constant. Default 0.5
            Noise : float in range [0,1]
                The amount of random noise to be added to the interferograms.
                Noise is measured as the variance (sigma) as a fraction over parameter B (Noise = sigma/B).
            phase_offset : float
                Initial phase offset (radians).

        Returns:
            I : np.array(Nx,Ny,Nz)
                Stack of interferograms
        """

        opd = Z[:, :, None] - scanning_steps[None, None, :]

        I = A + B*np.exp(
            -4*((opd/Lc)**2)
        ) * np.cos(
            phase_offset + (4*np.pi*opd)/lambda0
        )

        sigma = noise*B
        return I + np.random.normal(scale=sigma,size=I.shape)

    @staticmethod
    def PSI(Z, scanning_steps, A=0.5, B=0.5, noise=0, phase_offset=0):
        """
        Generates a PSI interferogram stack. Input height is in Radians.

        Parameters:
            Z : np.array(Nx,Ny)
                Input surface.
            scanning_steps : np.array(Nz)
                Array representing the height (or phase) of each scan.
            A : float
                Background illumnination constant.
            B : float
                Modulation contrast constant.
            Noise : float in range [0,1]
                The amount of random noise to be added to the interferograms.
                Noise is measured as the variance (sigma) as a fraction over parameter B (Noise = sigma/B).
            phase_offset : float
                Initial phase offset (radians).

        Returns:
            I : np.array(Nx,Ny,Nz)
                Stack of interferograms
        """

        phase = Z[:, :, None] + scanning_steps[None, None, :]

        I = A + B*np.cos(
                phase + phase_offset
        )
        sigma = noise*B

        nan_mask = np.isnan(Z)
        I[nan_mask,:] = A

        return I + np.random.normal(scale=sigma,size=I.shape)
