from __future__ import division
import os

import networkx as nx
import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
 RangeSet, SolverFactory, value
)

import pyomo.environ as pyo


class ShortestPathModel:
    """
    A class to set up and solve a shortest-path problem using Pyomo.

    This model finds a path from a specified origin to a specified destination
    with binary decision variables x[i, j] indicating whether the arc
    from i to j is used. The cost is minimized.
    """

    def __init__(self, 
                 num_nodes=5, 
                 large_constant=999, 
                 solver_name='glpk',
                 origin=1,
                 destination=None):
        """
        Initialize the shortest path model with node count, a large constant
        for prohibitive costs, a chosen solver name, and the origin/destination
        nodes for the path.

        Parameters
        ----------
        num_nodes : int, optional
            Number of nodes in the network (default is 5).
        large_constant : int or float, optional
            A large constant to represent a prohibitively large cost (default is 999).
        solver_name : str, optional
            Name of the solver to use (default is 'glpk').
        origin : int, optional
            The node from which the path should start (default is 1).
        destination : int, optional
            The node at which the path should end. If None (default),
            it is assumed to be `num_nodes`.
        """

        # Store parameters
        self.num_nodes = num_nodes
        self.large_constant = large_constant
        self.solver_name = solver_name
        self.origin = origin
        self.destination = destination if destination is not None else num_nodes

        # Placeholders for Pyomo objects
        self.model = None
        self.results = None

    def setup_model(self):
        """
        Set up the ConcreteModel, define sets, variables, cost, objective, and constraints.
        """
        # Clear console (optional, as in your original code)
        os.system("clear")

        # Create the model
        self.model = ConcreteModel()

        # Sets
        self.model.N = RangeSet(1, self.num_nodes)

        # Cost dictionary (example matches your original code)
        # You could also make this cost data an argument if you need more flexibility.
        self.model.cost = {
            (1, 1): 999, (1, 2): 5,   (1, 3): 2,   (1, 4): 999, (1, 5): 999,
            (2, 1): 999, (2, 2): 999, (2, 3): 999, (2, 4): 999, (2, 5): 8,
            (3, 1): 999, (3, 2): 999, (3, 3): 999, (3, 4): 3,   (3, 5): 999,
            (4, 1): 999, (4, 2): 999, (4, 3): 999, (4, 4): 999, (4, 5): 2,
            (5, 1): 999, (5, 2): 999, (5, 3): 999, (5, 4): 999, (5, 5): 999
        }

        # Decision variables: x[i,j] in {0,1}
        self.model.x = Var(self.model.N, self.model.N, domain=pyo.Binary)

        # Objective: minimize total cost
        def obj_rule(m):
            return sum(m.x[i, j] * m.cost[i, j] for i in m.N for j in m.N)
        self.model.obj = Objective(rule=obj_rule)

        # Constraints
        def source_rule(m, i):
            # The origin node must have exactly one outgoing arc
            if i == self.origin:
                return sum(m.x[i, j] for j in m.N) == 1
            else:
                return Constraint.Skip

        self.model.source = Constraint(self.model.N, rule=source_rule)

        def destination_rule(m, j):
            # The destination node must have exactly one incoming arc
            if j == self.destination:
                return sum(m.x[i, j] for i in m.N) == 1
            else:
                return Constraint.Skip

        self.model.destination = Constraint(self.model.N, rule=destination_rule)

        def intermediate_rule(m, i):
            # Flow balance for intermediate nodes
            # (excluding the origin and destination)
            if i != self.origin and i != self.destination:
                return (sum(m.x[i, j] for j in m.N)
                        - sum(m.x[j, i] for j in m.N)) == 0
            else:
                return Constraint.Skip

        self.model.intermediate = Constraint(self.model.N, rule=intermediate_rule)

    def solve_model(self):
        """
        Solve the model using the specified solver and store the results.

        Raises
        ------
        ValueError
            If the model is not set up before solving.
        """
        if self.model is None:
            raise ValueError("Model is not set up. Please call 'setup_model()' first.")

        # Create and invoke solver
        solver = SolverFactory(self.solver_name)
        self.results = solver.solve(self.model)

        return self.results

    def display_results(self):
        """
        Display the model's results, including variable values.
        """
        if not self.results:
            print("No results found. Please solve the model first.")
            return

        print("\n================= Pyomo Model Output =================")
        self.model.display()

        # Optional: Print arcs used in the solution
        print("\nArcs used in the solution (x[i, j] = 1):")
        used_edges = []
        for i in self.model.N:
            for j in self.model.N:
                if value(self.model.x[i, j]) == 1:
                    used_edges.append((i, j))
                    print(f"x[{i},{j}] = 1")

        if not used_edges:
            print("No edges were used. Possibly no valid path was found.")

    def plot_solution(self):
        """
        Plot the network graph using networkx. 
        Any arc with cost < large_constant is considered feasible. 
        The edges selected in the solution are highlighted in red.
        """
        if self.model is None:
            print("Model is not set up. Please call 'setup_model()' first.")
            return
        if not self.results:
            print("No solution to plot. Please solve the model first.")
            return

        # Build a directed graph based on the cost dictionary
        G = nx.DiGraph()
        for (i, j), c in self.model.cost.items():
            # Only add edges that are not 'prohibitively large'
            if c < self.large_constant:
                G.add_edge(i, j, weight=c)

        # Identify used edges based on x[i,j] = 1
        used_edges = [(i, j) 
                      for i in self.model.N 
                      for j in self.model.N 
                      if value(self.model.x[i, j]) == 1]

        # Create a layout for the nodes
        pos = nx.spring_layout(G, seed=42)  # 'seed' for reproducible layout

        # Draw the full graph in gray
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.4)

        # Add edge labels (costs)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Highlight the used edges in red
        if used_edges:
            nx.draw_networkx_edges(G, pos, edgelist=used_edges, edge_color='red', 
                                   arrows=True, width=2)
        else:
            print("Warning: No edges were used in the solution. Nothing to highlight.")

        # Show the plot
        plt.title(f"Shortest Path from {self.origin} to {self.destination}")
        plt.axis('off')
        plt.show()


# -----------------------------------------------------------------------------
# Example of how to use this class
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of the ShortestPathModel
    # By default, origin=1 and destination=last node => 5
    sp_model = ShortestPathModel(num_nodes=5, large_constant=999, solver_name='glpk')

    # Alternatively, you could do something like this if you wanted to go
    # from node 2 to node 4:
    # sp_model = ShortestPathModel(num_nodes=5, solver_name='glpk', origin=2, destination=4)

    # Set up the model
    sp_model.setup_model()

    # Solve the model
    results = sp_model.solve_model()

    # Display the results in text form
    sp_model.display_results()

    # Plot the solution using networkx
    sp_model.plot_solution()
