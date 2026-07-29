import numpy as np

class Generate:
    """
    The Surface.Generate module is a collection of methods that generate
    (simulate) different kinds of surfaces.
    """
    
    @staticmethod
    def Planar(Nx=64,
               Ny=None,
               tilt_direction=0,
               height=1,
               offset=0
               ):
        """
        Returns a planar surface. Surfaces Z-values always range from
        (0,height).

        Parameters:
            Nx : int
                Number of pixels in the X-dimension. Default is 64.
            Ny : int *optional*
                Number of pixels in the Y-Dimenion. If Ny is None,
                Ny will be set to Nx.
            tilt_direction : int
                Angle in degrees [0,360) that controls which edge of the part is highest (i.e., the tilt direction).
                Common values:
                - 0 → right edge is highest
                - 90 → top edge is highest
                - 180 → left edge is highest
                - 270 → bottom edge is highest
                Default: 0
            offset : float
                Constant offset added to the height of the surface.
            height : float
                The overall height of the surface. Default: 1

        Returns:
            Z : np.array(Nx,Ny)
                The resulting map of surface height.

        """

        if Ny is None:
            Ny = Nx

        x = np.linspace(0,1,Nx)
        y = np.linspace(0,1,Ny)

        X,Y = np.meshgrid(x,y,indexing='xy')
        tilt_direction = tilt_direction * np.pi/180
        tilt_direction *= -1
        Z = np.cos(tilt_direction)*X + np.sin(tilt_direction)*Y

        # Bring the surface to [0,1]
        Z -= Z.min()

        # Scale it to the desired height:
        Z *= height

        return Z + offset
        
    @staticmethod
    def Paraboloid(Nx=64,
               Ny=None,
               convex=True,
               height=1,
               offset=0
               ):
        """
        Returns a paraboloid surface. Surfaces Z-values always range from
        (0,height).

        Parameters:
            Nx : int
                Number of pixels in the X-dimension. Default is 64.
            Ny : int *optional*
                Number of pixels in the Y-Dimenion. If Ny is None,
                Ny will be set to Nx.
            convex : bool
                Determines whether the surface curves inward [False] (concave),
                or outward [True] (convex).
            height : float
                The overall height of the surface. Default: 1
            offset : float
                Constant offset added to the height of the surface.

        Returns
            Z : np.array(Nx,Ny)
                The resulting map of surface height.
        """

        if Ny is None:
            Ny = Nx

        v = np.sqrt(1/2)
        x = np.linspace(-v,v,Nx)
        y = np.linspace(-v,v,Ny)

        X,Y = np.meshgrid(x,y,indexing='xy')

        if convex:
            Z = 1 - (X**2) - (Y**2)
        else:
            Z = (X**2) + (Y**2)

        # Bring the surface to [0,1]
        Z -= Z.min()

        # Scale it to the desired height:
        Z *= height

        return Z + offset

    @staticmethod
    def Peaks(N=49,
        height=1,
        offset=0,
        ):
        """
        Returns a MatLab-like Peaks surface. Surfaces Z-values always scaled from
        (0,height).

        Parameters:
            N : int
                Number of pixels in the X- and Y-Dimensions. Default is 49.
            height : float
                The overall height of the surface. Default: 1
            offset : float
                Constant offset added to the height of the surface.

        Returns
            Z : np.array(Nx,Ny)
                The resulting map of surface height.
        """

        x = y = np.linspace(-3,3,N)

        X,Y = np.meshgrid(x,y,indexing='xy')

        # Create the Peaks Surface:
        Z = 3 * (1 - X)**2 * np.exp(-(X**2) - (Y + 1)**2) \
            - 10 * (X / 5 - X**3 - Y**5) * np.exp(-X**2 - Y**2) \
            - 1/3 * np.exp(-(X + 1)**2 - Y**2)

        # Bring the surface to [0,1]
        Z -= Z.min()
        Z /= Z.max()

        # Scale it to the desired height:
        Z *= height

        return Z + offset
    
    @staticmethod
    def Gaussian(Nx=64,
               Ny=None,
               convex=True,
               x0=None,
               y0=None,
               height=1,
               sigma=32,
               offset=0
               ):
        """
        Returns a gaussian surface. Surfaces Z-values always range from
        (0,height).

        Parameters:
            Nx : int
                Number of pixels in the X-dimension. Default is 64.
            Ny : int *optional*
                Number of pixels in the Y-Dimenion. If Ny is None,
                Ny will be set to Nx.
            convex : bool
                Determines whether the surface curves inward [False] (concave),
                or outward [True] (convex).
            height : float
                The overall height of the surface. Default: 1
            offset : float
                Constant offset added to the height of the surface.

        Returns
            Z : np.array(Nx,Ny)
                The resulting map of surface height.
        """

        if Ny is None:
            Ny = Nx

        if x0 is None:
            x0 = Nx//2
        if y0 is None:
            y0 = Ny//2

        x, y = np.arange(Nx), np.arange(Ny)
        X,Y = np.meshgrid(x,y,indexing='xy')

        Z = 1 - np.exp(-(X-x0)**2 / (sigma**2)) * np.exp(-(Y-y0)**2 / (sigma**2))
        # Bring the surface to [0,1]
        Z -= Z.min()

        # Scale it to the desired height:
        Z *= height

        return Z + offset
