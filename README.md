# Deep Q-learning breakout implementation

## Overview
This project implements a Deep Q-Network (DQN) agent that learns to play the classic Atari game Breakout. The implementation was created with the help of Claude 3.5 Sonnet and features a dual network architecture based on the 2015 DeepMind Nature DQN paper by Volodymyr Mnih et al, https://www.nature.com/articles/nature14236. This version has several optimisations for learning and execution efficiency on a single-core CPU running Windows OS.

![Breakout](https://github.com/user-attachments/assets/ccb88723-85ad-4c06-96d9-3dd72eaf9a26)


## Features
- Interactive training and gameplay visualisation using R Shiny
- Dual network (policy/target) architecture for more stable learning
- Greedy Epsilon approach to maximise exploration initially
- Multi-level reward system for efficient learning
- Simple save/load functionality for trained models
- Configurable hyperparameters
- Watch mode to observe trained agent

## Tech stack
- Python 3.12
- TensorFlow 2.17.0
- Numpy 1.26.4
- R 4.4.1 ("Race for Your Life")
- Shiny 1.9.1
- Reticulate 1.39.0 (for Python-R integration)

## Installation

### Prerequisites
```bash
# Python dependencies
pip install tensorflow numpy
```

```r
# R dependencies
install.packages(c("shiny", "reticulate", "grid"))
```

### Setup
1. Set up a new project in R
2. Clone the repository to your new project folder
3. Set up Python environment using a command window
```bash
python -m venv myenv
myenv\Scripts\activate
```
4. Run the application:
```r
shiny::runApp()
```

## How it works

### Core architecture

#### State representation
The agent observes 7 key state parameters:
- Paddle position (0 to 1)
- Ball X and Y positions
- Ball X and Y velocities
- Remaining bricks (as a proportion of the original total)
- Distance between the paddle and ball on the x-axis

#### Action space
The DQN agent can take three possible actions:
- Move left (0)
- Stay still (1)
- Move right (2)

#### Neural network architecture
Both policy and target networks use identical architecture:
```python
Sequential([
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(256, activation='relu'),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dense(3, activation='linear')
])
```

### Learning process

#### Dual network system
1. **Policy network**
   - This network is the active decision maker
   - Updated frequently through training
   - Learns from experience

2. **Target network**
   - This network provides stable learning targets for the policy network
   - Updated every 10 episodes
   - Prevents moving target problem

#### Experience collection
Stores experiences as (state, action, reward, next_state, done) tuples in a replay buffer:
- Buffer size: 10,000 experiences
- Training batch size: 64
- Minimum experiences before training: 128

#### Reward system
Unlike the original DeepMind DQN approach, this implementation uses a multi-level reward structure:
1. **Immediate rewards**
   - Breaking bricks
   - Multiple-hit combos
   - Paddle positioning

2. **Penalties**
   - Missing the ball
   - Game over (but the penalty is reduced if the agent has hit lots of bricks)

3. **Milestone rewards**
   - Level 1 completion
   - Level 2 completion

#### Learning algorithm
Implements Q-learning with experience replay:
```python
Q(s,a) = R + γ * max(Q(s',a'))
```
Where:
- Q(s,a): Action value
- R: Immediate reward
- γ: Discount factor (0.95)
- s': Next state
- a': Possible next actions

### Training process

#### Exploration strategy
Uses ε-greedy exploration:
- Initial ε: 1.0 (100% random actions)
- Final ε: 0.1 (10% random actions)
- Decay rate: 0.999

#### Training loop
1. Collect experience:
   - Observe state
   - Select action (ε-greedy)
   - Execute action
   - Store experience

2. Train networks:
   - Sample random batch
   - Compute target Q-values
   - Update policy network
   - Periodically update target network

## UI components
Implemented in Shiny. Although not great for visualising gameplay, it does allow good user control.

### Training controls
- Start/Stop training
- Reset agent
- Save/Load model
- Watch AI play
- Adjust game speed

### Metrics display
- Current game stats
- Training metrics
- Learning progress
- Network updates

## Results
After 100-200 episodes, the agent typically learns to:
- Successfully track and hit the ball
- Break bricks efficiently
- Complete levels
- Handle increasing difficulty

### After 2 episodes
All actions are random, so any block hits are 'accidental'.
![Breakout_init](https://github.com/user-attachments/assets/e2f90c22-2812-45ac-9478-d52071baac3e)

### After 20+ episodes
The agent has learnt to hit blocks occasionally.
![Breakout_20](https://github.com/user-attachments/assets/4f32639a-8c7b-48f9-aca7-25ff8916f081)

### After 150+ episodes
The agent breaks bricks efficiently.
![Breakout_150](https://github.com/user-attachments/assets/0254d9e6-4e3d-4b82-90d0-85984ccae5f3)

### After 170+ episodes
If allowed to play without exploration, the agent only very rarely misses the ball. The agent has learned to 'slice' the ball so that the angle of the hits on the blocks is very shallow so that a combination of blocks is likely to be hit in one go. This is not as effective as the 'tunnelling' strategy developed by the DeepMind DQN agent, but it does result in levels being completed.
![Breakout_170](https://github.com/user-attachments/assets/0955ff5c-a0cc-4625-984c-cbd893d0fd7d)

## Potential future improvements
I'm considering a few potential enhancements:

### Prioritised experience replay
Instead of randomly sampling experiences, assign higher sampling probability to important or rare experiences. Importance determined by TD error magnitude - larger errors indicate more surprising/valuable experiences. Would help agent learn more efficiently from critical moments like brick-breaking combos or near-misses.

### Dueling DQN architecture 
Split network into two streams:
1. Value stream: Estimates value of being in current state
2. Advantage stream: Estimates advantage of each action relative to others

Particularly beneficial for Breakout where some states are inherently valuable (ball heading toward bricks) regardless of chosen action. Could improve learning speed and stability.

### Rainbow DQN features
Combine multiple DQN improvements:
- Double Q-learning: Reduces value overestimation
- Noisy Networks: More sophisticated exploration than ε-greedy
- Distributional RL: Learn distribution of returns instead of just mean
- Multi-step learning: Use actual returns from multiple steps ahead

Would provide more robust learning but requires careful tuning of multiple components.

### Multi-threaded training
Current implementation runs on single CPU core. Parallelisation opportunities:
- Multiple game instances generating experiences simultaneously
- Parallel batch processing during network updates
- Asynchronous advantage actor-critic (A3C) architecture

Could significantly speed up training while maintaining learning quality.

### CNN-based state processing
Replace current state vector with raw game pixels processed by convolutional layers. In other words, implement the DeepMind version. Benefits:
- Learn directly from visual input like human players
- Could discover patterns/strategies not obvious in simplified state
- More generalisable to visual variants of game

Requires more computational resources but closer to original DQN breakthrough.

### Curiosity-driven exploration
Add intrinsic motivation through curiosity:
- Network predicts next state from current state-action
- Prediction errors generate intrinsic rewards
- Encourages exploration of unfamiliar situations

Could help discover advanced strategies like optimal brick-breaking sequences.

### Hierarchical DQN
Implement multiple layers of policies:
- Meta-controller: Selects high-level goals (clear specific areas, aim for combos)
- Controller: Learns basic actions to achieve selected goals
Could develop more sophisticated strategies through temporal abstraction.

### Human preference learning
Allow human feedback to shape reward function:
- Collect preferences between pairs of trajectories
- Learn reward function from human choices
- Could capture subtle aspects of "good" gameplay beyond score

Would help align agent behavior with human intuition of skilled play, although I'd expect a decent DQN agent to be able to outplay most humans if trained sufficiently.

## License
MIT License

## Acknowledgments
- DeepMind's DQN papers: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf and https://www.nature.com/articles/nature14236
- Inspiration from Stanford University students: https://cs.stanford.edu/~rpryzant/data/rl/paper.pdf
- Original Atari Breakout game
- TensorFlow and R Shiny communities
