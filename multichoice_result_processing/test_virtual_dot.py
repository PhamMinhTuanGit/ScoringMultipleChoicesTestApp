from visualization import visualize_batches
from grid_info import linearForwardDot

# Test the linearForwardDot function and visualize the results
if __name__ == "__main__":
    dotA = (1, 1)
    dotB = (4, 5)
    newdot = linearForwardDot(dotA, dotB, k=0.5)
    print("New dot:", newdot)

    # Create batches for visualization
    batches = {
        0: [dotA, dotB, newdot]
    }

    # Visualize the batches
    visualize_batches(batches)
