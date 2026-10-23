import numpy as np
from scipy.stats import norm
from models import BlackScholesModel
from payoffs import EuropeanOption

class MonteCarloEngine:
    def __init__(self, model, option):
        self.model = model
        self.option = option
        
    def price(self, num_paths: int, seed: int = 67):
        S_T = self.model.generate_terminal_prices(self.option.maturity, num_paths, seed)
        payoffs = self.option.payoff(S_T)
        
        # Discounted payoffs: e^{-rT} * E[Payoff]
        discount_factor = np.exp(-self.model.rate * self.option.maturity)
        discounted_payoffs = discount_factor * payoffs
        
        price_estimate = np.mean(discounted_payoffs)
        variance = np.var(discounted_payoffs, ddof=1)
        standard_error = np.sqrt(variance / num_paths)
        
        return price_estimate, standard_error, variance

def black_scholes_analytical(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


if __name__ == "__main__":
    # Parameters from Arouna (2004) Table 4.1
    spot = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    num_paths = 50000
    
    model = BlackScholesModel(spot, rate, vol)
    
    for strike in [80, 100, 120]:
        call_option = EuropeanOption(strike, maturity, 'call')
        engine = MonteCarloEngine(model, call_option)
        
        mc_price, mc_err, mc_var = engine.price(num_paths)
        exact_price = black_scholes_analytical(spot, strike, maturity, rate, vol, 'call')
        
        print(f"Strike: {strike}")
        print(f"MC Price: {mc_price:.4f} (Err: {mc_err:.4f})")
        print(f"Exact BS: {exact_price:.4f}")
        print("-" * 30)