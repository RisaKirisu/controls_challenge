### Gradualy introduce noises
- Train model by gradually introducing noises.
- Randomly use different types of noises to avoid model being trained to a specific type of noise.

### Use smoother reward function (more robust to noise)
- Reward can be noisy with control noise
- Use smoother reward
- Trace moving trajectory rather than directly using lat accel?

### Alternative action space
- Use change in steering rather than position of steering as action
- Use rate of change in steering rather than position of steering

### Add termination state for simulator
- Instead of continue the sim until input runs out, we terminate the sim when trajectory deviated too much from target.
- Look at where is the agent getting rewards. Maybe agent is getting away with randomly getting the output close enough