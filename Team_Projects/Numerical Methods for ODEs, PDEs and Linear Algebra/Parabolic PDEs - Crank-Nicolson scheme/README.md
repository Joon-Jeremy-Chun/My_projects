# Solar Pond Temperature Simulation

## Project Overview

This project involves simulating the temperature distribution in a solar pond using the Crank-Nicolson numerical scheme. A solar pond is a saltwater pool designed to collect and store solar energy. The presence of a salt gradient within the pond reduces heat convection, thereby minimizing heat loss and enhancing heat transfer via conduction.

## Problem Statement

The temperature \( u(x, t) \) at depth \( x \) and time \( t \) in the pond follows the heat equation, modified to include linear heat addition with depth, expressed as:

$$
u_t = u_{xx} + C(L−x)
$$

where:
- \( L \) is the depth of the pond,
- \( C \) is the heating rate,
- \( T0 \) is the initial surface water temperature.

Boundary conditions assume no heat transfer at the pond's bottom and maintain a constant surface temperature.

## Methodology

1. **Crank-Nicolson Scheme:** This numerical method approximates the temperature distribution, using a central difference method for the derivative boundary condition.
2. **Simulation Parameters:**
   - \( C = 0.5 \)
   - \( L = 1 \) (depth of the pond)
   - \( T0 = 60 \) (surface temperature)
   - Grid sizes: \( \Delta x = 0.1 \) and \( \Delta t = 0.005 \)
3. **Initial and Boundary Conditions:** These conditions define the initial temperature distribution and the boundary conditions at the pond’s surface and bottom.
4. **Tridiagonal Matrix Approach:** Employed for efficient resolution of the numerical scheme.

## Code Implementation

- Setup of domain and grid sizes.
- Definition of boundary and initial conditions.
- Construction of tridiagonal matrices and vectors for iterative solutions.
- Execution of iterative steps using the NumPy library to solve the system of equations.

## Results

The simulation provides temperature distributions at times \( t = 0 \), \( t = 0.5 \), and \( t = 1 \). These results are visualized in a single plot with different colors and legends for clarity.

## Conclusion

This project enhances the understanding of the Crank-Nicolson scheme, illustrating its application in solving linear equations. The results may be compared with those obtained using the Jacobi Method. Additionally, it provides insights into how effectively a solar pond can store thermal energy and how the temperature distribution changes over time under specified conditions.