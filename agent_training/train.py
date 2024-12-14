import copy

def train_agent(env, agent, episodes=400, print_every=10):
    rewards_history = []
    portfolio_history = []
    best_reward = -float('inf')
    best_state_dict = None

    
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train()
            
            state = next_state
            total_reward += reward
            
            if done:
                portfolio_history.append(info['portfolio_value'])
                
        rewards_history.append(total_reward)
        
        if (episode + 1) % print_every == 0:
            print(f"Episode: {episode + 1}, Reward: {total_reward:.2f}, "
                  f"Portfolio Value: ${info['portfolio_value']:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")
            
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if total_reward > best_reward:
            best_reward = total_reward
            print('New best reward!', 'Portfolio Value:', info['portfolio_value'], 'Total Reward:', total_reward)
            best_state_dict = copy.deepcopy(agent.policy_net.state_dict())
    
    return rewards_history, portfolio_history, best_state_dict
