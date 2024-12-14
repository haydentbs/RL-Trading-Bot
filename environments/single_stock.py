import gym
import numpy as np
import pandas as pd
from gym import spaces

class SingleStockEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(SingleStockEnv, self).__init__()
        
        self.df = df
        
        self.current_step = 0

        self.initial_price = df.iloc[0]['Close']

        self.initial_portfolio_value = initial_balance
        
        # Action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.max_shares_held = initial_balance / self.initial_price

        self.initial_shares_held = int(initial_balance / (2*self.initial_price))
        self.initial_balance = initial_balance - self.initial_shares_held * self.initial_price

        self.trading_rate = 0.2
        self.trading_cost = 0.003
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = self.initial_shares_held
        self.cost_basis = 0
        self.total_trades = 1
        self.total_profit = 0
        return self._get_observation()
    
    def _get_observation(self):
        # Get current price data
        current_price = self.df.iloc[self.current_step]['Close']
        
        
        # Calculate technical indicators
        rsi = self.df.iloc[self.current_step]['RSI']
        ma_ratio = self.df.iloc[self.current_step]['MA_ratio']
        volatility = self.df.iloc[self.current_step]['Volatility']
        
        # Portfolio info
        portfolio_value = self.balance + (self.shares_held * current_price)
        profit_pct = (portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value

        self.max_shares_held = portfolio_value / current_price
        
        # Create observation as a numpy array
        obs = np.array([
            current_price / self.df['Close'].mean() - 1,  # Normalized price
            self.balance / self.initial_portfolio_value - 1,  # Normalized balance
            self.shares_held * current_price / self.initial_portfolio_value,  # Position size
            profit_pct,
            rsi / 100 - 0.5,  # Normalized RSI
            ma_ratio - 1,  # MA ratio
            volatility,
            self.total_trades / 100  # Normalized trade count
        ])
        
        # Ensure the observation is a numpy array of type float32
        return obs.astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        initial_portfolio_value = self.balance + (self.shares_held * current_price)
        
        # Execute trade
        reward = 0

        max_shares = int(initial_portfolio_value / (current_price))
        shares_to_trade = max_shares * self.trading_rate

        if action == 0:  # Sell
            if shares_to_trade < self.shares_held:
                # Sell all shares
                
                self.balance += shares_to_trade * current_price * (1 - self.trading_cost)  # 0.3% transaction cost
                reward = (current_price - self.cost_basis) * self.shares_held / self.initial_portfolio_value
                self.shares_held = self.shares_held - shares_to_trade
                self.total_trades += 1
            else:
                reward -= 1
            if self.total_trades > self.current_step / 3:  # More than 1 trade per 10 steps
                reward -= 5 * np.log(self.total_trades/ (self.current_step / 3))
                
        elif action == 2:  # Buy
            if self.balance > shares_to_trade * current_price:
                # Calculate maximum shares we can buy
                if shares_to_trade > 0:
                    self.shares_held += shares_to_trade
                    self.balance -= shares_to_trade * current_price * (1 + self.trading_cost)
                    self.cost_basis = current_price
                    self.total_trades += 1
                else:
                    reward -= 1
            if self.total_trades > self.current_step / 3:  # More than 1 trade per 10 steps
                reward -= 5 * np.log(self.total_trades/ (self.current_step / 3))
        
        # Calculate portfolio value and return
        portfolio_value = self.balance + (self.shares_held * current_price)
        profit = portfolio_value - self.initial_portfolio_value
        self.total_profit = profit
        
        # Calculate reward
        reward += profit / self.initial_portfolio_value
        
        # Add trading frequency penalty
        
        
        info = {
            'portfolio_value': portfolio_value,
            'profit': profit,
            'trades': self.total_trades,
            'shares_held': self.shares_held,
            'max_shares_held': self.max_shares_held
        }
        
        return self._get_observation(), reward, done, info