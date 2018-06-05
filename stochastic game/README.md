# I3 in stochastic games

### Games envs
- zero-sum: matching pennies  (zero_sum)
- prisoner dilemma   (prisoner)
- nash equilibrium    (nash)
- cooperative     (cooperative)

### Agents 
I3
Q

### run scripts
python3 I3.py --game NAME_OF_GAME --agent AGENT_TYPE 

### Other Arguments
- timestep: the timestep for action trajectories, default 2
- iterations: the number of iterations of training, default 60000
- steps: the number of steps in one interation, default 1000
- seed: random seed
- batch_size: off-policy training batch size, defualt 200 
- warm_up_steps: the number of iterations to perform warm start, deault 1000
