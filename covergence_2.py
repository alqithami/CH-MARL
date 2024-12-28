import numpy as np
import matplotlib.pyplot as plt

# Environment Parameters
resource_cap = 100  # Total resource cap (high-level constraint)
num_agents = 5  # Number of low-level agents
fairness_threshold = 0.8  # Fairness threshold
episodes = 3000  # Further extended training episodes for convergence
high_level_learning_rate = 0.0005  # Learning rate for high-level policy
low_level_learning_rate = 0.01  # Learning rate for low-level policies
initial_fairness_weight = 1.0  # Initial fairness penalty weight
max_fairness_weight = 5.0  # Maximum fairness penalty weight
baseline_reward = 20  # Baseline reward for agents
smoothing_factor = 0.9  # Smoothing factor for regularizing high-level allocations
fairness_smoothing_window = 50  # Window size for rolling average smoothing of fairness metric

# High-Level Policy Initialization
high_level_policy = np.random.rand(10) * 10  # High-level allocation decisions
smoothed_allocation = 0  # Variable to track smoothed high-level allocations

# Low-Level Policies Initialization
low_level_policies = [np.random.rand(10) for _ in range(num_agents)]  # Individual agent policies

# Tracking Metrics
high_level_allocations = []  # Total high-level resource allocations
low_level_rewards = []  # Low-level agent rewards
fairness_metrics = []  # Fairness metric
resource_usages = []  # Total resource usage
reward_std_devs = []  # Standard deviation of rewards across agents (fairness proxy)

# Training Loop
for episode in range(episodes):
    # Scale fairness penalty weight over time
    fairness_weight = initial_fairness_weight + (
        (max_fairness_weight - initial_fairness_weight) * episode / episodes
    )

    # High-Level Policy Decision (Allocate Resources with Smoothing)
    raw_allocation = high_level_policy[np.random.randint(0, 10)] * 10
    total_allocation = smoothing_factor * smoothed_allocation + (1 - smoothing_factor) * raw_allocation
    total_allocation = min(total_allocation, resource_cap)  # Ensure allocation doesn't exceed cap
    smoothed_allocation = total_allocation  # Update smoothed allocation
    high_level_allocations.append(total_allocation)

    # Low-Level Agent Actions
    allocations = [policy[np.random.randint(0, 10)] * (total_allocation / num_agents) for policy in low_level_policies]
    total_usage = sum(allocations)
    
    # Check Constraints
    if total_usage > resource_cap:
        penalties = [0.5 * (total_usage - resource_cap) for _ in allocations]  # Stronger penalty for exceeding cap
    else:
        penalties = [0 for _ in allocations]

    # Compute Rewards and Fairness
    agent_rewards = [baseline_reward - alloc - pen for alloc, pen in zip(allocations, penalties)]
    reward_std_dev = np.std(agent_rewards)  # Standard deviation of rewards for fairness assessment
    reward_std_devs.append(reward_std_dev)
    fairness_penalty = fairness_weight * reward_std_dev  # Penalize variance in rewards
    agent_rewards = [r - fairness_penalty for r in agent_rewards]  # Apply fairness penalty

    # Add explicit fairness reward term (reducing reward variance)
    fairness_reward = -reward_std_dev  # Reward agents collectively for minimizing variance
    agent_rewards = [r + fairness_reward for r in agent_rewards]

    max_reward = max(agent_rewards) if max(agent_rewards) > 0 else 1
    fairness = min(agent_rewards) / max_reward  # Fairness metric
    fairness = max(fairness, 0)  # Ensure fairness is non-negative
    
    # Log Metrics
    low_level_rewards.append(agent_rewards)
    fairness_metrics.append(fairness)
    resource_usages.append(total_usage)

    # Update Policies
    high_level_policy += high_level_learning_rate * (np.random.rand(10) - 0.5)  # High-level policy perturbation
    for i in range(num_agents):
        low_level_policies[i] += low_level_learning_rate * (np.random.rand(10) - 0.5)  # Low-level policy perturbation

# Apply rolling average to fairness metrics for smoothing
smoothed_fairness_metrics = np.convolve(fairness_metrics, np.ones(fairness_smoothing_window)/fairness_smoothing_window, mode='valid')

# Convert Results to Numpy Arrays for Analysis
low_level_rewards = np.array(low_level_rewards)
fairness_metrics = np.array(fairness_metrics)
resource_usages = np.array(resource_usages)
high_level_allocations = np.array(high_level_allocations)
reward_std_devs = np.array(reward_std_devs)

# Plot Results
plt.figure(figsize=(12, 12))

# Plot High-Level Allocations
plt.subplot(5, 1, 1)
plt.plot(high_level_allocations, label="High-Level Allocations")
plt.axhline(y=resource_cap, color='r', linestyle='--', label="Resource Cap")
plt.xlabel("Episode")
plt.ylabel("Allocation")
plt.title("High-Level Allocations Over Episodes")
plt.legend()

# Plot Total Resource Usage
plt.subplot(5, 1, 2)
plt.plot(resource_usages, label="Total Resource Usage")
plt.axhline(y=resource_cap, color='r', linestyle='--', label="Resource Cap")
plt.xlabel("Episode")
plt.ylabel("Resource Usage")
plt.title("Total Resource Usage Over Episodes")
plt.legend()

# Plot Mean Rewards for Low-Level Policies
plt.subplot(5, 1, 3)
plt.plot(np.mean(low_level_rewards, axis=1), label="Mean Low-Level Reward")
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Low-Level Agent Rewards Over Episodes")
plt.legend()

# Plot Smoothed Fairness Metrics
plt.subplot(5, 1, 4)
plt.plot(smoothed_fairness_metrics, label="Smoothed Fairness Metric")
plt.axhline(y=fairness_threshold, color='r', linestyle='--', label="Fairness Threshold")
plt.xlabel("Episode")
plt.ylabel("Fairness Metric")
plt.title("Fairness Metrics Over Episodes (Smoothed)")
plt.legend()

# Plot Reward Standard Deviations
plt.subplot(5, 1, 5)
plt.plot(reward_std_devs, label="Reward Standard Deviation")
plt.xlabel("Episode")
plt.ylabel("Reward Std Dev")
plt.title("Reward Standard Deviations Over Episodes (Fairness Proxy)")
plt.legend()

plt.tight_layout()

# Save the figure
output_path = "results/CHMARL_Refined_Fairness.png"
plt.savefig(output_path, dpi=300)
print(f"Figure saved at: {output_path}")

# Show the plot
plt.show()
