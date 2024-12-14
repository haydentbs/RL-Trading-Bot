import gym
import numpy as np
import pandas as pd
from gym import spaces

class MultiStockEnv(gym.Env):
    def __init__(self, data_dict, initial_balance=100000):
        super(MultiStockEnv, self).__init__()
        
        self.data_dict = data_dict
        self.symbols = list(data_dict.keys())
        self.n_stocks = len(self.symbols)
        
        self.current_step = 0
        self.initial_balance = initial_balance
        
        # Action space: for each stock: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = spaces.MultiDiscrete([3] * self.n_stocks)
        
        # Observation space: price data + account info for each stock
        # 8 features per stock + balance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(8 * self.n_stocks + 1,), 
            dtype=np.float32)
        
        self.initial_prices = {symbol: data_dict[symbol].iloc[0]['Close'] 
                             for symbol in self.symbols}
        
        self.trading_cost = 0.003
        
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        
        # Initialize holdings for each stock
        self.shares_held = {symbol: 0 for symbol in self.symbols}
        self.cost_basis = {symbol: 0 for symbol in self.symbols}
        self.total_trades = {symbol: 0 for symbol in self.symbols}
        
        return self._get_observation()
    
    def _get_observation(self):
        obs = []
        
        for symbol in self.symbols:
            df = self.data_dict[symbol]
            current_price = df.iloc[self.current_step]['Close']
            
            # Calculate stock-specific features
            rsi = df.iloc[self.current_step]['RSI']
            ma_ratio = df.iloc[self.current_step]['MA_ratio']
            volatility = df.iloc[self.current_step]['Volatility']
            
            portfolio_value = self.balance + sum(
                [self.shares_held[s] * self.data_dict[s].iloc[self.current_step]['Close'] 
                 for s in self.symbols])
            
            stock_obs = [
                current_price / df['Close'].mean() - 1,
                self.shares_held[symbol] * current_price / self.initial_balance,
                rsi / 100 - 0.5,
                ma_ratio - 1,
                volatility,
                self.total_trades[symbol] / 100,
                self.shares_held[symbol],
                self.cost_basis[symbol] / current_price - 1 if self.cost_basis[symbol] > 0 else 0
            ]
            obs.extend(stock_obs)

        # Add overall balance
        obs.append(self.balance / self.initial_balance - 1)
        
        return np.array(obs, dtype=np.float32)
        
    def step(self, actions):
        # Initialize reward and info
        total_reward = 0
        portfolio_value_before = self.balance + sum(
            [self.shares_held[s] * self.data_dict[s].iloc[self.current_step]['Close'] 
            for s in self.symbols])
        
        symbol = self.symbols[actions // self.n_stocks]
        action = actions % self.n_stocks

        # Process actions for each stock
        # for i, symbol in enumerate(self.symbols):
        current_price = self.data_dict[symbol].iloc[self.current_step]['Close']
            # print(actions)
            # action = actions[i]

            # Calculate tradeable shares
        max_shares = int(self.balance / current_price)
        shares_to_trade = int(max_shares * 0.2)  # Trade 20% of max possible

        if action == 0:  # Sell
            if self.shares_held[symbol] > 0:
                # Sell shares
                shares_to_sell = min(shares_to_trade, self.shares_held[symbol])
                sell_value = shares_to_sell * current_price * (1 - self.trading_cost)
                self.balance += sell_value
                self.shares_held[symbol] -= shares_to_sell
                self.total_trades[symbol] += 1
                
                # Calculate reward based on profit
                profit = (current_price - self.cost_basis[symbol]) * shares_to_sell
                total_reward += profit / self.initial_balance
            else:
                total_reward -= 0.1  # Penalty for invalid sell

        elif action == 2:  # Buy
            if self.balance > shares_to_trade * current_price:
                # Buy shares
                purchase_cost = shares_to_trade * current_price * (1 + self.trading_cost)
                if purchase_cost <= self.balance:
                    self.balance -= purchase_cost
                    self.shares_held[symbol] += shares_to_trade
                    self.cost_basis[symbol] = current_price
                    self.total_trades[symbol] += 1
            else:
                total_reward -= 0.1  # Penalty for invalid buy

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(next(iter(self.data_dict.values()))) - 1

        # Calculate final portfolio value and return
        portfolio_value_after = self.balance + sum(
            [self.shares_held[s] * self.data_dict[s].iloc[self.current_step]['Close'] 
            for s in self.symbols])

        # Add portfolio return to reward
        portfolio_return = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        total_reward += portfolio_return

        # Add trading frequency penalty
        for symbol in self.symbols:
            if self.total_trades[symbol] > self.current_step / 3:
                total_reward -= 0.1 * np.log(self.total_trades[symbol] / (self.current_step / 3))

        info = {
            'portfolio_value': portfolio_value_after,
            'balance': self.balance,
            'shares': self.shares_held,
            'trades': self.total_trades,
        }

        return self._get_observation(), total_reward, done, info
                
        