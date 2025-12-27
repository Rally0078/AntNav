## A (not so) simple Desert Ant navigation simulation

The animation is done using PyBullet. For simplicity, I've just used a default visual model for representing the agent(s).

While the real desert ant can go back to its nest based on the sun's position as a reference, there is an assumption made here that the sun's position is the same as the nest's position. This is of course not true in the real world, but enough for a simple simulation. In reality, the ant may lose track of its nest after traveling too far, because it may not remember all of its positions due to the sun position changing, or it may even run out of memory to remember its path relative to the sun in some way. Such limitations to the dead reckoning navigation are not implemented here.

The agent and environment are built with reinforcement learning in mind, so the environment subclasses `gym.Env`. In theory, it should be possible to use Stable Baselines3 to train the agent, which I've included already in the conda environment as a pip package. But this requires some additional observation(s) such as an olfactory or vision sense that the agent can use. Otherwise, any RL training will train a random policy which is practically useless, since the food cubes themselves are randomly spawned.

### Installation:

You need conda for this.
```bash
    chmod +x setup_deps.sh
    ./setup_deps.sh
```

### Running:

Simply run
```bash
conda activate ant_navigation
python3 main.py
```