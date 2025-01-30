from pyomo.environ import ConcreteModel, Var, Objective, Constraint, maximize
from pyomo.opt import SolverFactory
import pyomo.environ as pyo

class LinearOptimizationModel:
    """
    A class to set up and solve a simple linear optimization problem using Pyomo.
    
    Attributes:
    -----------
    solver_name : str
        The name of the solver to be used (default: 'glpk').
    model : ConcreteModel or None
        The Pyomo ConcreteModel instance. None before setup.
    results : SolverResults or None
        The results from solving the model. None before solve.
    """

    def __init__(self, solver_name='glpk'):
        """
        Initialize the model class with a chosen solver name.
        
        Parameters:
        -----------
        solver_name : str, optional
            The solver to use for solving the linear program (default is 'glpk').
        """
        self.solver_name = solver_name
        self.model = None
        self.results = None

    def setup_model(self):
        """
        Set up the Pyomo model with variables, objective function, and constraints.
        """
        # Create a Pyomo ConcreteModel
        self.model = ConcreteModel()

        # Define variables
        self.model.x = Var([1, 2], domain=pyo.NonNegativeReals)

        # Define objective function: maximize 3*x1 + 2*x2
        self.model.obj = Objective(
            expr=3 * self.model.x[1] + 2 * self.model.x[2],
            sense=maximize
        )

        # Define constraints
        self.model.res1 = Constraint(expr=2 * self.model.x[1] + self.model.x[2] <= 100)
        self.model.res2 = Constraint(expr=self.model.x[1] + self.model.x[2] <= 80)
        self.model.res3 = Constraint(expr=self.model.x[1] <= 40)

    def solve_model(self):
        """
        Solve the model using the specified solver, store, and return the results.
        
        Raises:
        -------
        ValueError
            If the model is not set up before attempting to solve.
        """
        if self.model is None:
            raise ValueError("Model is not set up. Please call 'setup_model()' before solving.")

        # Create the solver instance
        solver = SolverFactory(self.solver_name)
        # Solve the model
        self.results = solver.solve(self.model)

        return self.results

    def display_results(self):
        """
        Display the model's results and decision variable values.
        
        If no results are available, prints a warning message.
        """
        if self.results is None:
            print("No results to display. Please solve the model first.")
        else:
            self.model.display()


# Example usage
if __name__ == "__main__":
    # Create an instance of the optimization model class with default solver
    linear_model = LinearOptimizationModel()

    # Set up the model
    linear_model.setup_model()

    # Solve the model
    solve_results = linear_model.solve_model()

    # Display the results
    linear_model.display_results()
