import numpy as np

class EuropeanOption:
    def __init__(self, strike: float, maturity: float, option_type: str = 'call'):
        self.strike = strike # K
        self.maturity = maturity # T
        self.option_type = option_type.lower()
        
    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        if self.option_type == 'call':
            return np.maximum(S_T - self.strike, 0.0)
        elif self.option_type == 'put':
            return np.maximum(self.strike - S_T, 0.0)
        else:
            raise ValueError("Type must be 'call' or 'put'")