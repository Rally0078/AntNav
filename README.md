## A (not so) simple Desert Ant navigation simulation

The animation is done using PyBullet. For simplicity, I've just used a default visual model for representing the agent.

### Installation:

You need conda for this.
```bash
    chmod +x setup_deps.sh
```

The agent and environment are built with reinforcement learning in mind, so the environment subclasses `gym.Env`. In theory, it should be possible to use Stable Baselines3 to train the agent, which I've included already in the conda environment as a pip package.

### Running:

Simply run
```bash
conda activate ant_navigation
python3 main.py
```