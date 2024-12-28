# CH-MARL: Constrained Hierarchical Multi-Agent Reinforcement Learning

## Overview
This repository contains the implementation and theoretical exploration of the Constrained Hierarchical Multi-Agent Reinforcement Learning (CH-MARL) framework. CH-MARL is designed to optimize decision-making in multi-agent systems while satisfying global constraints, such as emissions caps, and promoting fairness across agents.

Key features of CH-MARL:
- **Hierarchical Structure:** High-level policies define strategic goals, while low-level policies optimize local decisions under constraints.
- **Constraint Satisfaction:** Ensures global constraints, such as resource usage or emissions caps, are respected.
- **Fairness Mechanism:** Promotes equitable outcomes among agents using fairness-aware reward structures.
- **Scalability:** Efficiently handles large-scale environments and complex agent interactions.

## Features
- **Theoretical Contributions:** Convergence guarantees, bounded constraint satisfaction, and fairness metrics.
- **Hierarchical Policy Optimization:** A two-level framework for strategic and local decision-making.
- **Computational Analysis:** Detailed complexity analysis of CH-MARL's time and space requirements.

## Repository Structure
- `src/`: Contains the Python scripts for training and testing the CH-MARL framework.
- `theory/`: Includes detailed proofs of the theoretical propositions, including convergence guarantees and fairness metrics. (TBA)
- `results/`: Output data and visualizations from experiments. (TBA)
- `README.md`: Project description and usage instructions.
- `requirements.txt`: List of dependencies for running the project.

## Getting Started
### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/alqithami/CH-MARL.git
   cd CH-MARL
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments
To train and validate the CH-MARL framework:
```bash
python src/chmarl.py
```
Generated results, including fairness metrics and resource utilization graphs, will be saved in the `results/` directory.

## Theoretical Insights
### Key Propositions
1. **Convergence to Constraint-Satisfying Policies:** Ensures policies meet global constraints through Lagrangian optimization.
2. **Bounded Constraint Violations:** Guarantees deviations from constraints remain within acceptable thresholds.
3. **Fairness Metrics:** Ensures fairness among agents based on a max-min fairness criterion.

### Complexity Analysis
- **Time Complexity:**
  - High-level policy: \( O(T \cdot |A^H| \cdot |S|) \)
  - Low-level policies: \( O(n \cdot T \cdot |A^L|) \)
  - Overall: \( O(T \cdot |A^H| \cdot |S| + n \cdot T \cdot |A^L|) \)
- **Space Complexity:** \( O(|S| \cdot |A^H| + n \cdot |S| \cdot |A^L|) \)

## Future Work
Future directions for CH-MARL include:
- Enhancing partial observability models with graph neural networks and attention mechanisms.
- Developing adversarial and fault-tolerant approaches.
- Conducting real-world pilot studies in maritime logistics and beyond.

## References
- Foerster, J. et al. (2016). Learning to communicate with deep multi-agent reinforcement learning.
- Lin, L. et al. (2021). Hierarchical reinforcement learning in multi-agent settings.
- Shi, H. et al. (2020). Multi-agent coordination under global constraints.

## License
This project is licensed for non-commercial academic use only. For commercial applications, contact the authors for licensing terms.

## Acknowledgments
Special thanks to the contributors and collaborators who made this work possible.
