""" 
Contains classes that build the mesh! 
"""
import numpy as np

class Mesh: 
    """ 
    Abstract base class which builds 1D or 2D mesh 
    """
    @staticmethod
    def build_from_config(cfg : dict): 
        dim = cfg['mesh']['dim']
        if dim == 1: 
            return Mesh1D(cfg) 
        elif dim == 2: 
            return Mesh2D(cfg)
        else: 
            raise ValueError(f"Unsupported dimension: {dim}")


class Mesh1D(Mesh): 
    """ 
    Builds 1D mesh 
    """
    def __init__(self, cfg : dict): 
        self.L = cfg['mesh']['L']
        self.nCells = cfg['mesh']['nCells']
        self.dx = self.L / self.nCells
        self.x = np.linspace(0, self.L, self.nCells, endpoint = False)

        # Setting up periodic indices 
        if cfg['mesh']['periodic_bc']: 
            self.idx = np.arange(self.nCells)
            self.idxRightNeighb = (self.idx + 1) % self.nCells # Helps with finite differences  
            self.idxLeftNeighb = (self.idx - 1) % self.nCells 
        else: 
            raise ValueError("Cannot handle non-periodic boundary conditions")

class Mesh2D(Mesh): 
    """ 
    Builds 2D mesh with optional terrain and velocity field
    """
    def __init__(self, cfg : dict): 
        self.Lx = cfg['mesh']['Lx']
        self.Ly = cfg['mesh']['Ly']
        self.nCellsX = cfg['mesh']['nCellsX']
        self.nCellsY = cfg['mesh']['nCellsY']
        self.dx = self.Lx / self.nCellsX
        self.dy = self.Ly / self.nCellsY

        self.x = np.linspace(0, self.Lx, self.nCellsX, endpoint = False)
        self.y = np.linspace(0, self.Ly, self.nCellsY, endpoint = False)
        self.xGrid, self.yGrid = np.meshgrid(self.x,self.y, indexing='xy')

        if cfg['mesh']['periodic_bc']: 
            # Setting up periodic indices 
            self.idxX = np.arange(self.nCellsX)
            self.idxRightNeighb = (self.idxX + 1) % self.nCellsX
            self.idxLeftNeighb = (self.idxX - 1) % self.nCellsX

            self.idxY = np.arange(self.nCellsY)
            self.idxTopNeighb = (self.idxY + 1) % self.nCellsY
            self.idxBottomNeighb = (self.idxY - 1) % self.nCellsY
        else: 
            raise ValueError("Cannot handle non-periodic boundary conditions")
        
        # Terrain and velocity fields (initialized as None, set by terrain builder)
        self.h = None
        self.vx = None
        self.vy = None
    
    def set_terrain_and_velocity(self, h: np.ndarray, vx: np.ndarray, vy: np.ndarray):
        """
        Set terrain and velocity fields for this mesh.
        
        Inputs:
            h (np.ndarray): Terrain height field, shape (nCellsY, nCellsX)
            vx (np.ndarray): x-velocity field, shape (nCellsY, nCellsX)
            vy (np.ndarray): y-velocity field, shape (nCellsY, nCellsX)
        """
        assert h.shape == (self.nCellsY, self.nCellsX), f"Terrain shape mismatch: got {h.shape}, expected ({self.nCellsY}, {self.nCellsX})"
        assert vx.shape == (self.nCellsY, self.nCellsX), f"vx shape mismatch"
        assert vy.shape == (self.nCellsY, self.nCellsX), f"vy shape mismatch"
        
        self.h = h
        self.vx = vx
        self.vy = vy

    










