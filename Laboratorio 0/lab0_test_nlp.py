# -*- coding: utf-8 -*-

from pyomo.environ import (
    ConcreteModel, Var, Objective, PositiveReals,
    minimize, SolverFactory, value
)

class MinSumDistanceModel:
    """
    A class to set up and solve a Pyomo model that minimizes
    the sum of weighted distances to a set of points (demand points).

    Attributes
    ----------
    A : list or iterable
        The set (or list) of point identifiers.
    x_coords : dict
        Dictionary of x-coordinates for each point in A.
    y_coords : dict
        Dictionary of y-coordinates for each point in A.
    demand : dict
        Dictionary of demands (weights) for each point in A.
    solver_name : str
        Name of the solver to be used (e.g., 'ipopt').
    model : ConcreteModel
        The Pyomo model object (initialized to None until setup).
    results : SolverResults
        The solver results object (initialized to None until solve).
    """

    def __init__(self, 
                 A=None, 
                 x_coords=None, 
                 y_coords=None, 
                 demand=None, 
                 solver_name='ipopt'):
        """
        Initialize the model with coordinate and demand data, plus the solver name.

        Parameters
        ----------
        A : list, optional
            Identifiers of points. Defaults to [1,2,3,4] if None.
        x_coords : dict, optional
            X-coordinates of points. Defaults to {1:2, 2:6, 3:2, 4:6} if None.
        y_coords : dict, optional
            Y-coordinates of points. Defaults to {1:1, 2:1, 3:5, 4:5} if None.
        demand : dict, optional
            Demand (weights) at each point. Defaults to {1:100, 2:200, 3:300, 4:400} if None.
        solver_name : str, optional
            The solver to use (default: 'ipopt').
        """
        # Default data
        if A is None:
            A = [1, 2, 3, 4]
        if x_coords is None:
            x_coords = {1: 2, 2: 6, 3: 2, 4: 6}
        if y_coords is None:
            y_coords = {1: 1, 2: 1, 3: 5, 4: 5}
        if demand is None:
            demand = {1: 100, 2: 200, 3: 300, 4: 400}

        self.A = A
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.demand = demand
        self.solver_name = solver_name

        # Pyomo model placeholders
        self.model = None
        self.results = None

    def setup_model(self):
        """
        Set up the ConcreteModel, define variables, objective, and constraints (if any).
        """
        self.model = ConcreteModel()

        # Decision variables (location coordinates)
        self.model.x = Var(domain=PositiveReals)
        self.model.y = Var(domain=PositiveReals)

        # Objective: minimize sum of (demand[i] * distance from (x,y) to (x_coords[i],y_coords[i]))
        def objective_rule(m):
            return sum(
                self.demand[i] * ((self.x_coords[i] - m.x)**2 + (self.y_coords[i] - m.y)**2) ** 0.5
                for i in self.A
            )

        self.model.g = Objective(rule=objective_rule, sense=minimize)

    def solve_model(self):
        """
        Solve the model using the specified solver, store the results, and return them.

        Raises
        ------
        ValueError
            If the model is not set up before solving.
        """
        if self.model is None:
            raise ValueError("Model is not set up. Please call 'setup_model()' first.")

        solver = SolverFactory(self.solver_name)
        self.results = solver.solve(self.model)

        return self.results

    def display_results(self):
        """
        Display the model's results, including the chosen (x, y) location and the objective value.
        """
        if not self.results:
            print("No results found. Please solve the model first.")
            return

        self.model.display()
        print("\nOptimal location (x, y) = ({:.4f}, {:.4f})".format(
            value(self.model.x), value(self.model.y)
        ))
        print("Objective value (min sum of weighted distances): {:.4f}".format(
            value(self.model.g)
        ))


# -----------------------------------------------------------------------------
# Example of how to use this class
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an instance of the model (with default data and 'ipopt' solver).
    min_sum_dist_model = MinSumDistanceModel()

    # Set up the model
    min_sum_dist_model.setup_model()

    # Solve the model
    results = min_sum_dist_model.solve_model()

    # Display the results
    min_sum_dist_model.display_results()
