# Boundary Value Problem & the Jacobi Method

## Overview

This project focuses on solving a boundary value problem (BVP) associated with the deflection of a thin beam supported at both ends under a uniform load. The deflection, denoted as `u(x)`, varies along the beam and is governed by a specific differential equation with given boundary conditions.

The core of the project involves discretizing the BVP into a system of linear equations and then applying the Jacobi method to approximate the solution. This process converts the continuous model into a discrete form suitable for computational analysis.

## Differential Equation

The deflection of the beam `u(x)` is described by the differential equation:



d^2u/dx^2 = -q(x) / EI



- Where:

- `d^2u/dx^2` is the second derivative of the deflection `u` with respect to the position `x` along the beam.
- `q(x)` represents the load applied per unit length as a function of `x`.
- `E` is the modulus of elasticity of the beam material.
- `I` is the moment of inertia of the beam's cross-section.

*Note: Replace the equation above with the specific equation from your document.*

## Objectives

- **Discretization of the BVP**: Transform the continuous BVP into a discrete system of linear equations using the central difference approximation for the second-order derivative.
- **System Construction**: Construct a tridiagonal system of linear equations derived from the discretization process with a specified mesh size `h`.
- **Jacobi Method Implementation**: Implement the Jacobi iterative method to solve the resulting system of linear equations, thereby approximating the beam's deflection at specified points.
- **Analysis and Visualization**: Analyze the convergence of the Jacobi method and visualize the deflection of the beam across its length.

## Methodology

1. **Discretization**: The BVP is first discretized using the central difference approximation, resulting in a tridiagonal matrix representation of the problem.
2. **System of Linear Equations**: With a specified mesh size `h`, the problem is transformed into a system of linear equations characterized by a tridiagonal matrix.
3. **Jacobi Method**: The Jacobi iterative method is then employed to approximate the solution to the system. The method's convergence is monitored through an error metric, with iterations continuing until the error falls below a predetermined threshold or a maximum number of iterations is reached.
4. **Visualization**: The approximate solution is plotted to visualize the beam's deflection, providing insights into the physical behavior of the system under the given load.

## Expected Outcome

The project aims to efficiently approximate the deflection `u(x)` of the beam using the Jacobi method. The number of iterations required to achieve the desired accuracy and the maximum deflection observed will be key outcomes of this analysis. Additionally, a plot of the deflection across the beam will be provided to illustrate the solution's physical implications.