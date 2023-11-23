# An implementation of the NSGA-II evolutionary algorithm for multi-objective optimization.

# References
# [1] *Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan*, |A Fast
# Elitist Multiobjective Genetic Algorithm: NSGA-II|, IEEE Transactions on
# Evolutionary Computation 6 (2002), no. 2, 182 ~ 197.

# [2] *N. Srinivas and Kalyanmoy Deb*, |Multiobjective Optimization Using
# Nondominated Sorting in Genetic Algorithms|, Evolutionary Computation 2
# (1994), no. 3, 221 ~ 248.

import numpy as np


# Initializes population of solutions
def init_pop(pop_size, num_obj, obj_weights, normalize, num_DV, min_vals,
             max_vals):

  # Populate decision variable array with random values within range
  decision_vals = np.random.uniform(low=min_vals,
                                    high=max_vals,
                                    size=(pop_size, num_DV))

  # Evaluate population using decision variables for each objective function
  objectives = eval_pop(decision_vals, num_obj, obj_weights, normalize)

  # Concatenate values so second axis of population array is all decision var and evaluated objective function values
  P = np.concatenate((decision_vals, objectives), axis=1)
  P = P.tolist()

  # Return population of solutions in array size pop_size x (num_DV + num_obj)
  return P


# Evaluate objective functions of solution given decision variable values and exogenous function of the objective functions
def eval_obj(decision_vars, num_obj, obj_weights, normalize):
  # If no objective weights included, weight all equally
  if obj_weights == None:
    obj_weights = np.ones((num_obj))

  # NOTE: OBJ_FUNCT must evaluate each objective, with # functions = # objectives, selecting pertinent decision variables for each from the passed values
  obj_vals = OBJ_FUNCT(decision_vars, num_obj)

  if normalize is True:
    # Produce normalized function values for each candidate to compare solutions using sum of performance values
    mean = np.mean(obj_vals)
    stdev = np.std(obj_vals)

    # Subtract mean, divide by standard deviation, of values
    obj_vals_normalized = (obj_vals - mean) / stdev

    # Assign weights to different objectives
    obj_vals_weighted = obj_vals_normalized * obj_weights

  # Weight objectives
  obj_vals_weighted = obj_vals * obj_weights

  # Returns weighted and potentially normalized objective function values
  return obj_vals_weighted


# Evaluates population of solution decision variable values to produce objective function values to compare fitness
def eval_pop(decision_vars, num_obj, obj_weights, normalize):
  pop_size, num_DV = decision_vars.shape

  # Initialize objective function array of dims pop size x number objectives
  obj_functions = np.zeros((pop_size, num_obj))

  # Evaluate objective functions for each individual and objective given DV values
  for i in range(pop_size):
    obj_functions[i, :] = eval_obj(decision_vars[i, :], num_obj, obj_weights,
                                   normalize)

  # Create list of all solutions' objective performance of dimensions pop_size x num_obj
  objectives = obj_functions.tolist()

  return objectives


# Sorts population into non-dominated Pareto front with assocated ranks
# Returns sorted Pareto front and list of corresponding solution ranks
def sort_pop(P, num_obj):
  # Extract population size and number of objective function (performance values) from each solution
  pop_size = len(P)
  num_DV = len(P[0]) - num_obj
  total_dims = len(P[0])

  # Pareto front set F (the sorted Pareto sets)
  F = []
  # Sorting set S
  S = []
  # Rank set (the rank of each solution)
  rank = []

  # Domination counter (how many times a solution is dominated)
  n = [0] * pop_size

  # Front counter, starts at first
  f = 1

  # For each individual in the population, determine degree of domination and populate first front (nondominated)
  for i in range(pop_size):
    # Extract solution
    p = P[i]
    # Initialize set dominated by p
    S.append([])
    # Initialize p domination counter
    n[i] = 0
    # For each individual in the population
    for j in range(pop_size):
      # Extract other solution
      q = P[j]
      # If p dominates q, objective function values only
      if p is not q and sum(p[num_DV:total_dims]) < sum(q[num_DV:total_dims]):
        # Then add q to dominated set S_p
        S[i].append(q)
      # Otherwise increment p domination counter
      elif p is not q and sum(p[num_DV:total_dims]) > sum(
          q[num_DV:total_dims]):
        n[i] += 1
    # If p not dominated by any other solution
    if n[i] == 0:
      # Add p to front
      F.append(p)
      # Assign rank to p (lower = better, 1 = best, non-dominated)
      rank.append(1)

  # Fill front with solutions assuming there are dominated solutions left to consider
  while sum(n) > 0:
    # Parse each individual in given front
    for i in range(len(F)):
      # Select solution from front
      p = F[i]
      # Find index of front solution in population
      b = P.index(p)

      # Search set of solutions dominated by p
      for j in range(len(S[b])):
        # Select individual solution q from set
        q = S[b][j]
        # Find index of dominated solution in original population
        d = P.index(q)
        # Decrement domination counter
        n[d] -= 1
        # Determine if solution worthy of inclusion in new front
        if n[d] == 0:
          # Add q to next front
          F.append(q)
          # Assign rank of solution q
          rank.append(f + 1)

    # Increment the front counter
    f += 1

  # Return sorted Pareto front (2d list) and rank corresponding to each member of front (list)
  return F, rank


# Assigns distances for each solution in particular front to gauge and ultimately promote diversity in solutions space
def crowding_dist(F, num_obj):
  # Number of individuals and objective values in front
  front_size = len(F)
  num_DV = len(F[0]) - num_obj

  # For each i, set I(i)distance = 0, initialize distance
  dists = np.zeros((front_size))
  dist = np.zeros((front_size))

  # Convert list to Numpy array for easier numerical manipulation
  F = np.asarray(F)

  # For each objective in set:
  for obj in range(0, num_obj):
    # Sort front using each objective value
    I = np.sort(F[:, num_DV + obj], axis=0)

    # Find minimum and maximum values for objective function outputs
    max_val = np.max(I)
    min_val = np.min(I)

    # Calculate all crowding distances within front
    for i in range(1, front_size - 1):
      # Find individual value in sorted 1d objective front
      m = I[i]
      # Locate index of value in original front
      j = np.where(F[:, num_DV + obj] == m)

      # Add distance between neighbors normalized by range of distances in front
      dists[j] += (I[i + 1] - I[i - 1]) / (max_val - min_val)

    # Boundary points (extrema) have infinite distance to encourage their selection
    dists[np.where(F[:, num_DV + obj] == max_val)] += np.inf
    dists[np.where(F[:, num_DV + obj] == min_val)] += np.inf

  distances = dists.tolist()

  # Return distance array
  return distances


# Sorts Pareto optimal front to ensure more fit and diverse solutions given rank and crowding distance
def sort_front(F, rank, dists):
  # Initialize new front
  F_new = []
  front_size = len(F)

  # Parse through front and compare all solutions pairwise, adding better solution to front
  for i in range(len(F)):
    for j in reversed(range(len(F))):
      # Make sure solutions are different
      if i is not j:
        # Prefer solution with better (lower) rank...
        if (rank[i] < rank[j]) and (F[i] not in F_new):
          F_new.append(F[i])
        elif (rank[j] < rank[i]) and (F[j] not in F_new):
          F_new.append(F[j])
        # ... or less crowded region if rank is equal
        elif (rank[i] is rank[j] and dists[i] > dists[j]) and (F[i]
                                                               not in F_new):
          F_new.append(F[i])
        elif (rank[i] is rank[j] and dists[j] > dists[i]) and (F[j]
                                                               not in F_new):
          F_new.append(F[j])

  # Any stragglers in sorting are added last
  for x in F:
    if x not in F_new:
      F_new.append(x)

  # Return sorted front
  return F_new


# Produces offspring from parent population using reproductive processes of crossover and mutation
# Inputs: array of decision variable and evaluated objective values for all solutions in parent population, number of objectives, weights,
# whether objectives are normalized, number of decision variables, minimum and maximum DV values,
# probability and eta parameter for Simulated Binary Crossover, probability and eta parameter for polynomial mutation
def reproduction(parents, num_obj, obj_weights, normalize, num_DV, min_vals,
                 max_vals, sbx_prob, sbx_param, mutat_prob, mutat_param):

  # Population size
  pop_size = len(parents)

  # Crossover and mutation process flags
  was_crossover = None
  was_mutation = None

  # Array for storing next generation
  offspring = []

  # Initialize child population count
  c = 0

  # Until child population produced:
  while c < pop_size:

    # Perform SBX crossover with given probability for each decision variable of individuals
    a = np.random.random()
    if a < sbx_prob:
      # Initialize two children
      child1_vals = []
      child2_vals = []

      # Randomly select parents
      rand = np.random.choice(pop_size - 1, size=3, replace=False)

      # Get decision variable values of each parent
      parent1_vals = parents[rand[0]][:num_DV]
      parent2_vals = parents[rand[1]][:num_DV]

      # Assert difference between parents
      if parent1_vals == parent2_vals:
        parent2_vals = parents[rand[2]][:num_DV]

      # Perform simulated binary crossover (Deb & Agrawal 1995) for each decision variable of parents
      u = np.random.random()
      if u < 0.5:
        beta = (2 * u)**(1 / (sbx_param + 1))
      else:
        beta = (1 / (2 * (1 - u)))**(1 / (sbx_param + 1))

      # Produce offspring DV values
      # Ensure all child DV values are within defined acceptable ranges
      for v in range(num_DV):
        child1_vals.append(0.5 * (((1 + beta) * parent1_vals[v]) +
                                  ((1 - beta) * parent2_vals[v])))
        child2_vals.append(0.5 * (((1 - beta) * parent1_vals[v]) +
                                  ((1 + beta) * parent2_vals[v])))

        # If value outside range, set to respective extremum
        if child1_vals[v] > max_vals[v]:
          child1_vals[v] = max_vals[v]
        if child2_vals[v] > max_vals[v]:
          child2_vals[v] = max_vals[v]
        if child1_vals[v] < min_vals[v]:
          child1_vals[v] = min_vals[v]
        if child2_vals[v] < min_vals[v]:
          child2_vals[v] = min_vals[v]

      # Evaluate objective functions for offspring
      child1_obj = eval_obj(child1_vals, num_obj, obj_weights, normalize)
      child2_obj = eval_obj(child2_vals, num_obj, obj_weights, normalize)

      # Concatenate DV values and objective results to create children
      if num_DV == 1:
        child1_vals.extend(child1_obj)
        child1 = child1_vals
        child2_vals.extend(child2_obj)
        child2 = child1_vals
      else:
        child1 = child1_vals + child1_obj
        child2 = child2_vals + child2_obj

      # Activate crossover flag
      was_crossover = True
      was_mutation = False

    # Perform polynomial mutation
    b = np.random.random()
    if b < mutat_prob:

      # Initialize one child
      child3_vals = []

      # Generate parent
      rand = np.random.choice(pop_size - 1, size=1, replace=False)

      # Get parent values
      parent3_vals = parents[rand[0]][:num_DV]

      # Perform mutation on each parent value
      for w in range(num_DV):
        r = np.random.random()
        if r < 0.5:
          delta = (2 * r)**(1 / (1 + mutat_param)) - 1
        else:
          delta = 1 - (2 * (1 - r))**(1 / (1 + mutat_param))

        # Generate child element
        child3_vals.append(parent3_vals[w] + delta)

        # Assert values within decision space
        if child3_vals[w] > max_vals[w]:
          child3_vals[w] = max_vals[w]
        if child3_vals[w] < min_vals[w]:
          child3_vals[w] = min_vals[w]

      # Evaluate objective functions for offspring
      child3_obj = eval_obj(child3_vals, num_obj, num_DV)

      # Concatenate DV values and objective results to create child
      if num_DV == 1:
        child3_vals.extend(child3_obj)
        child3 = child3_vals
      else:
        child3 = child3_vals + child3_obj

      # Activate mutation flag
      was_mutation = True
      was_crossover = False

    # Fill offspring population with each of the generated children
    if was_crossover:
      offspring.append(child1)
      offspring.append(child2)
      was_crossover = None
      c += 2
    if was_mutation:
      offspring.append(child3)
      was_mutation = None
      c += 1
    if c == pop_size:
      break
    if c > pop_size:
      del (offspring[pop_size])
      break

  return offspring


# Non-Dominated Sorting Genetic Algorithm (NSGA-II) by Deb 2002
# Evaluates solutions comprising of decision variables to find optimal Pareto set of solutions for multiple objective functions
# Inputs: population size, number of generations, number of objectives, weights for each objective, number of decision variables,
# maximum DV values (upper and lower bounds of search space), minimum DV values,crossover probability and index parameter, mutation probability and index parameter.
# Assumes minimization problem
# Returns: initial and final (optimal) populations of solutions
def nsga(pop_size, num_gens, num_obj, obj_weights, num_DV, min_vals, max_vals,
         normalize, sbx_prob, sbx_param, mutat_prob, mutat_param):

  # Population array (ultimately of length num_gens), will contain arrays of solutions and objective function values
  P = []

  # Create initial parent population of solutions, arrays of dims pop_size x num_obj
  P_init = init_pop(pop_size, num_obj, obj_weights, normalize, num_DV,
                    min_vals, max_vals)
  P = P_init

  # Initial offspring population is empty until birthed
  Q = []

  # Initialize generation counter
  t = 0

  # Run generational cycle until limit reached
  while t < num_gens:
    # Combine parent, offspring populations
    R = P + Q

    # Find Pareto front within combined population i.e. minimum objective function values
    F, rank = sort_pop(R, num_obj)

    # Find distances within front
    dists = crowding_dist(F, num_obj)

    # Create next generation's parent population
    P_new = []

    # Sort Pareto front given rank and crowding distance
    F_new = sort_front(F, rank, dists)

    # Create parent population from new front with maximum fitness
    P_new = F_new[0:pop_size]

    # Reproduce population with selection, crossover, and mutation processes
    Q = reproduction(P_new, num_obj, obj_weights, normalize, num_DV, min_vals,
                     max_vals, sbx_prob, sbx_param, mutat_prob, mutat_param)

    # Count new generation
    t += 1

    # New pop is now old
    P = P_new

  # Return initial and final populations
  return P_init, P
