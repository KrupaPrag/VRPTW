# Vehicle Routing Problem with Time Windows
Computational Logistics of the Vehicle Routing Problem with Time Windows (VRPTW). Comparative Review of application of the solution techniques, the Particle Swarm Optimisation (PSO) algorithm and Genetic Algorithm (GA) to the VRPTW.

Conference Proceedings: https://ieeexplore.ieee.org/document/9004294

## SOLUTION TECHNIQUE ALGORITHMS
The solution technique algorithms are based and ccording to the respective references given below. The algorithms are coded using Python 3.

### Genetic Algorithm for the VRPTW:
Ombuki, Beatrice, Brian J. Ross, and Franklin Hanshar. "Multi-objective genetic algorithms for vehicle routing problem with time windows." Applied Intelligence 24.1 (2006): 17-30.
https://link.springer.com/content/pdf/10.1007/s10489-006-6926-z.pdf

#### Particle Swarm Optimisation algorithm for the VRPTW:
Gong YJ, Zhang J, Liu O, Huang RZ, Chung HS, Shi YH. Optimizing the vehicle routing problem with time windows: a discrete particle swarm optimization approach. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews). 2012 Mar;42(2):254-67.
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5773510

#### Datasets: Solomon Benchmarking Dataset
https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

#### Updated files:
For each of the algorithms and the respective metrics, the code for the initial encoding and optimisation can be found in the following files:
Initial Encoding: '/initial_encoding/main_initialEncoding.py'
Optimisation: '/main/main.py'

NOTE: Initial encoding needs to be run before running the main file in order to create the initial candidate solutions.

Please note that the '/main/final' folder contains some experiments from solving the VRPTW for datasets with 25 customers. This folder would need to be replicated to run the datasets with 50 and 100 customers.  
