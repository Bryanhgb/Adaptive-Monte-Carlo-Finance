import numpy as np

class BlackScholesModel:
    def __init__(self, spot: float, rate: float, vol: float):
        self.spot = spot  
        self.rate = rate  
        self.vol = vol    # sigma
        
    def generate_terminal_prices(self, T: float, num_paths: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        # X ~ N(0, 1) standard normal variables
        Z = np.random.standard_normal(num_paths)
        
        # S_T = x * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
        drift = (self.rate - 0.5 * self.vol**2) * T
        diffusion = self.vol * np.sqrt(T) * Z
        
        return self.spot * np.exp(drift + diffusion)

