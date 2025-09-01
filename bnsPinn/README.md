# Physics-informed neural networks for weakly compressible flows using Galerkin-Boltzmann formulation

Code accompanying the manuscript titled "Physics-informed neural networks for weakly compressible flows using Galerkin-Boltzmann formulation", authored by Atakan Aygun and Ali Karakus.

# Abstract

In this work, we study the Galerkin–Boltzmann formulation within a physics-informed neural network (PINN) framework to solve flow problems in weakly compressible regimes. The Galerkin–Boltzmann equations are discretized with second-order Hermite polynomials in microscopic velocity space, which leads to a first-order conservation law with six equations. Reducing the output dimension makes this equation system particularly well suited for PINNs compared with the widely used D2Q9 lattice Boltzmann velocity space discretizations. We created two distinct neural networks to overcome the scale disparity between the equilibrium and non-equilibrium states in collision terms of the equations. We test the accuracy and performance of the formulation with benchmark problems and solutions for forward and inverse problems with limited data. We compared our approach with the incompressible Navier–Stokes equation and the D2Q9 formulation. We show that the Galerkin–Boltzmann formulation results in similar L2 errors in velocity predictions in a comparable training time with the Navier–Stokes equation and lower training time than the D2Q9 formulation. We also solve forward and inverse problems for a flow over a square, try to capture an accurate boundary layer, and infer the relaxation time parameter using available data from a high-fidelity solver. Our findings show the potential of utilizing the Galerkin–Boltzmann formulation in PINN for weakly compressible flow problems.

# Citation

	@article{aygun_physics-informed_2024,
		title = {Physics-informed neural networks for weakly compressible flows using {Galerkin}–{Boltzmann} formulation},
		volume = {36},
		issn = {1070-6631},
		url = {https://doi.org/10.1063/5.0235756},
		doi = {10.1063/5.0235756},
		number = {11},
		urldate = {2025-01-06},
		journal = {Physics of Fluids},
		author = {Aygun, A. and Karakus, A.},
		month = nov,
		year = {2024},
		pages = {117125},
	}