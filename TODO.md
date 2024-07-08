# To Do

## File Info
### DQN
- `uav_env.py`: Environment
- `dqn.py`: DQN (NN, training function)
- `evaluate.py`: Evaluate policynet
- `visualize.py`: Visualize using plotly (uses output of `evaluate.py`)
- `random_policy.py`: Test policynet

Currently testing in `test.ipynb`.

### Other
These files are used to satisfy *Task 2.2.3: Visualize the coverage and movement of UAVs in the environment*. They may be redundant in the future with `visualize.py`
- `visual_matplotlib.py`: Visualize using matplotlib
- `visual_plotly.py`: Visualize using plotly


## Checklist
### In Progress
- [ ] `dqn.py`
    - [ ] Save outputs
    - [ ] Add replay memory (stabilize/improve DQN)
    - [ ] Make work?
    - [ ] Improve performance (currently creating tensor using list numpy arrays)

### Done
- [x] environment setup
- [x] working game environment, evaluation, and visualization (works with random policy)

### House-Keeping (unimportant save for last)
- [ ] add conditionals to setup to handle different versions of pytorch (CPU, GPU (+cu118), OS)
- [ ] clean up files
- [ ] make visualization prettier
- [ ] add 3D