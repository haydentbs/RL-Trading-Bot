import matplotlib.pyplot as plt
import numpy as np

def visualize_trades_with_benchmark(env, agent, df):
    # Run agent strategy
    state = env.reset()
    done = False
    
    buy_dates = []
    sell_dates = []
    portfolio_values = [env.balance]
    initial_balance = env.balance
    dates = df.index
    current_step = 0
    shares_held = []
    actions = [[], []]
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        if action == 2:  # Buy
            buy_dates.append(dates[current_step])
        elif action == 0:  # Sell
            sell_dates.append(dates[current_step])
            
        portfolio_values.append(info['portfolio_value'])
        shares_held.append(info['shares_held'] / (info['max_shares_held']) if shares_held else info['shares_held'])

        state = next_state
        current_step += 1
        actions[0].append(action)
        actions[1].append(0)  # Assuming single action dimension for simplicity
    
    shares_held[0] = 0.5
    
    # Calculate buy-and-hold strategy
    initial_stock_price = df['Close'].iloc[0]
    num_shares_buyhold = 2*initial_balance / initial_stock_price
    buyhold_values = df['Close'] * num_shares_buyhold 
    
    # Create subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), height_ratios=[1.5, 1, 1.5, 1])
    
    # Plot 1: Stock price with buy/sell points
    ax1.plot(dates, df['Close'], label='Stock Price', alpha=0.7)
    ax1.scatter(buy_dates, df.loc[buy_dates]['Close'], color='green', marker='^', label='Buy', s=100)
    ax1.scatter(sell_dates, df.loc[sell_dates]['Close'], color='red', marker='v', label='Sell', s=100)
    ax1.set_title('Stock Price with Trading Signals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Portfolio Value Comparison
    ax2.plot(dates[1:len(portfolio_values)], portfolio_values[1:], label='Agent Strategy', color='blue')
    ax2.plot(dates[1:], buyhold_values[1:], label='Buy and Hold Strategy', color='orange', alpha=0.7)
    ax2.set_title('Portfolio Value Comparison')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value ($)')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Stock Holdings
    ax3.plot(dates[:len(shares_held)], shares_held, label='Shares Held')
    ax3.set_title('Stock Holdings Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Stock Holdings')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Actions Taken
    for size in range(3):  # Assuming 3 action types: Sell, Hold, Buy
        mask = np.array(actions[0]) == size
        ax4.scatter(np.array(dates[:len(actions[0])])[mask], np.array(actions[0])[mask], label=f'Action {size}', alpha=0.7)
    
    ax4.set_title('Actions Taken Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Action Type')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Sell', 'Hold', 'Buy'])
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and print statistics
    agent_final_value = portfolio_values[-1]
    buyhold_final_value = buyhold_values.iloc[-1]
    
    agent_return = ((agent_final_value - initial_balance) / initial_balance) * 100
    buyhold_return = ((buyhold_final_value - initial_balance) / initial_balance) * 100
    
    print("\n=== Performance Comparison ===")
    print(f"Initial Investment: ${initial_balance:,.2f}")
    print("\nAgent Strategy:")
    print(f"Final Value: ${agent_final_value:,.2f}")
    print(f"Total Return: {agent_return:.2f}%")
    print(f"Number of Trades: {len(buy_dates) + len(sell_dates)}")
    
    print("\nBuy and Hold Strategy:")
    print(f"Final Value: ${buyhold_final_value:,.2f}")
    print(f"Total Return: {buyhold_return:.2f}%")
    
    # Calculate additional metrics
    agent_values = np.array(portfolio_values)
    agent_returns = np.diff(agent_values) / agent_values[:-1]
    buyhold_returns = buyhold_values.pct_change().dropna()
    
    agent_sharpe = np.sqrt(252) * np.mean(agent_returns) / np.std(agent_returns)
    buyhold_sharpe = np.sqrt(252) * np.mean(buyhold_returns) / np.std(buyhold_returns)
    
    # Maximum drawdown calculation
    agent_peak = np.maximum.accumulate(agent_values)
    agent_drawdown = (agent_peak - agent_values) / agent_peak
    agent_max_drawdown = np.max(agent_drawdown) * 100
    
    buyhold_peak = np.maximum.accumulate(buyhold_values)
    buyhold_drawdown = (buyhold_peak - buyhold_values) / buyhold_peak
    buyhold_max_drawdown = np.max(buyhold_drawdown) * 100
    
    print("\nRisk Metrics:")
    print(f"Agent Sharpe Ratio: {agent_sharpe:.2f}")
    print(f"Buy-Hold Sharpe Ratio: {buyhold_sharpe:.2f}")
    print(f"Agent Maximum Drawdown: {agent_max_drawdown:.2f}%")
    print(f"Buy-Hold Maximum Drawdown: {buyhold_max_drawdown:.2f}%")


def visualize_multi_stock_trades(env, agent, data_dict):
    state = env.reset()
    done = False
    
    portfolio_values = []
    actions_history = {symbol: [] for symbol in env.symbols}
    dates = list(data_dict[env.symbols[0]].index)
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        symbol = env.symbols[action // env.n_stocks]
        action_int = action % env.n_stocks
        
        # Record actions for each stock
        
        actions_history[symbol].append(action_int)
        for symbols in env.symbols:
            if symbols != symbol:
                actions_history[symbols].append(-1)
        
        portfolio_values.append(info['portfolio_value'])
        state = next_state
    
    # Create visualization
    fig, axes = plt.subplots(len(env.symbols) + 1, 1, figsize=(15, 5*len(env.symbols)))
    
    # Plot portfolio value
    axes[0].plot(dates[:len(portfolio_values)], portfolio_values)
    axes[0].set_title('Total Portfolio Value')
    
    # Plot individual stock prices and actions
    for i, symbol in enumerate(env.symbols):
        ax = axes[i+1]
        df = data_dict[symbol]
        ax.plot(dates, df['Close'], label=f'{symbol} Price')
        
        # Plot buy/sell points
        for j, action in enumerate(actions_history[symbol]):
            if action == 2:  # Buy
                ax.scatter(dates[j], df['Close'].iloc[j], color='green', marker='^')
            elif action == 0:  # Sell
                ax.scatter(dates[j], df['Close'].iloc[j], color='red', marker='v')
        
        ax.set_title(f'{symbol} Trading Activity')
        ax.legend()
    
    plt.tight_layout()
    plt.show()