"""
Created on Jan 19 11:07:58 2025

@author:
  - Gérman Montoya
  - Juan Andrés Mendez

A class-based script that demonstrates:
 1. How to check Python dependencies (numpy, pandas, matplotlib).
 2. How to draw a simple network with matplotlib.
 3. How to create and display a pandas DataFrame.
 4. How to use the math.ceil function in a demonstration.
 5. How to perform a small test in numpy.

Run:
  python your_file.py
"""

import importlib
import math

class NetworkPlotExample:
    """
    A class demonstrating how to:
    1. Check for missing dependencies (numpy, pandas, matplotlib).
    2. Draw a simple network (one or more points, potentially connected by an edge).
    3. Create and print a sample pandas DataFrame.
    4. Demonstrate usage of math.ceil (and other simple math).
    5. Perform a small numpy test (creating an array and computing stats).

    Attributes
    ----------
    dependencies : list of str
        A list of Python package names required by this demonstration code.
    """

    def __init__(self):
        """
        Initialize the demonstration class with the required dependencies
        as a list of strings.
        """
        self.dependencies = ["numpy", "pandas", "matplotlib"]

    def check_dependencies(self):
        """
        Check whether required dependencies are installed.
        If not, raise an ImportError with instructions on how to install them.
        """
        for dep in self.dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                raise ImportError(
                    f"Missing dependency '{dep}'. "
                    f"Please install it via 'pip install {dep}' or equivalent."
                )

    def numpy_test(self):
        """
        Perform a quick test with numpy to confirm it is installed and working.
        Creates an array and prints basic statistics.
        """
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print("\n--- NumPy Test ---")
        print("Array:", arr)
        print("Sum of array:", np.sum(arr))
        print("Mean of array:", np.mean(arr))
        print("-----------------")

    def draw_network(self):
        """
        Draw a flashier network with matplotlib, showcasing:
          - Multiple subplots in one figure
          - Legends for nodes and edges
          - Custom colors and styles
          - Double scales (secondary y-axis) in a second subplot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        # Use a different style if you want (e.g., 'seaborn', 'bmh', etc.)
        plt.style.use('ggplot')

        # Create a figure with 2 subplots side by side
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        # --------------------------------------------------------
        # Subplot 1: A simple network with labeled nodes and an edge
        # --------------------------------------------------------
        ax1 = axes[0]

        # Example coordinates for two nodes
        x1, y1 = 10, 30
        x2, y2 = 20, 40

        # Plot the first node
        ax1.plot(x1, y1, marker='o', color='blue', label='Node 1', markersize=8)
        ax1.text(x1 + 0.5, y1, "1", size=10)

        # Plot the second node
        ax1.plot(x2, y2, marker='o', color='green', label='Node 2', markersize=8)
        ax1.text(x2 + 0.5, y2, "2", size=10)

        # Draw an edge if the distance is below a threshold
        R_c = 15
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist <= R_c:
            ax1.plot([x1, x2], [y1, y2], 'r--', label='Edge')

        ax1.set_xlabel('X coordinates')
        ax1.set_ylabel('Y coordinates')
        ax1.set_title('Subplot 1: Simple Network')
        ax1.grid(True)
        ax1.legend(loc='best')

        # --------------------------------------------------------
        # Subplot 2: Another "network" with random points + double scale
        # --------------------------------------------------------
        ax2 = axes[1]

        # Generate random points for demonstration
        np.random.seed(42)  # For reproducible results
        X = np.random.randint(5, 25, size=5)
        Y = np.random.randint(10, 50, size=5)

        # Scatter plot of these points
        scatter = ax2.scatter(X, Y, c='purple', s=60, alpha=0.7, label='Random Nodes')
        # Label each point
        for i, (xx, yy) in enumerate(zip(X, Y), start=1):
            ax2.text(xx + 0.5, yy, str(i), fontsize=9)

        # Create a secondary y-axis for additional data
        ax2_b = ax2.twinx()
        # Example line plot on the secondary y-axis
        # We'll just use some artificial data here
        x_line = np.linspace(0, 10, 50)
        y_line = 1000 * np.sin(x_line) + 2000
        line2 = ax2_b.plot(x_line, y_line, color='orange', linewidth=2.0, label='Secondary Scale')

        ax2.set_xlabel('Random X')
        ax2.set_ylabel('Random Y', color='purple')
        ax2_b.set_ylabel('Secondary Scale', color='orange')

        ax2.set_title('Subplot 2: Random Points + Double Scale')
        ax2.grid(True)

        # Build legends for both the primary and secondary y-axes
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax2_b.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        # --------------------------------------------------------
        # Adjust layout and show the figure
        # --------------------------------------------------------
        fig.tight_layout()
        plt.show()

    def show_cars_dataframe(self):
        """
        Create a sample Cars DataFrame using pandas and print it.
        """
        import pandas as pd

        Cars = {
            'Brand': ['Honda Civic', 'Toyota Corolla', 'Ford Focus', 'Audi A4'],
            'Price': [22000, 25000, 27000, 35000]
        }

        df = pd.DataFrame(Cars, columns=['Brand', 'Price'])
        print(df)

    def math_demo(self):
        """
        Demonstrate usage of math.ceil (and any other math functions).
        """
        a1 = math.ceil(10.3)
        print(f"\nExample of math.ceil(10.3) -> {a1}")

    def run_example(self):
        """
        Run all demonstrations in order:
         1. Numpy test
         2. Draw a simple network.
         3. Show a sample DataFrame.
         4. Demonstrate math functions.
        """
        self.numpy_test()
        self.draw_network()
        self.show_cars_dataframe()
        self.math_demo()

if __name__ == "__main__":
    # Create an instance of the example class
    demo = NetworkPlotExample()

    # 1. Check dependencies
    try:
        demo.check_dependencies()
    except ImportError as e:
        print(e)
        print("Please install the missing package(s) and re-run.")
        exit(1)

    # 2. Run all demonstrations
    demo.run_example()
