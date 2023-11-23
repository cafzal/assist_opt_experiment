
## Purpose
Goal is to pair a domain expert with an AI assistant to optimize a network. 
1. Help define an optimization problem
2. Help select and run the optimization method to solve the problem
3. Help interpret and use solutions


### Example problem
Uses:
Supply_Chain_Example_Data.csv
nsga-ii.py

The decision variables for the multi-objective optimization problem are as follows:
- Location of the logistics centers (Tyler, Midland, Waco)
- Capacities of the logistics centers
- Transportation routes between the supply locations and the destinations (Los Angeles, Newark, Savannah)
- Inventory levels at the supply locations

The objectives are as follows:
- Minimize delays
- Minimize operational expenses

The constraints will include:
- Capital and operational budgets for the centers
- Capacity constraints at the logistics centers
- Demand satisfaction constraints
- Service level constraints
- Transportation mode constraints
