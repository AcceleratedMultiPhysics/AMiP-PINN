# Physics informed neural networks for two dimensional incompressible thermal convection problems

Code accompanying the manuscript titled "Physics informed neural networks for two dimensional incompressible thermal convection problems"

# Abstract

Physics-informed neural networks (PINNs) have drawn attention in recent years in engineering problems due to their effectiveness and ability to tackle problems without generating complex meshes. PINNs use automatic differentiation to evaluate differential operators in conservation laws and hence do not need a discretization scheme. Using this ability, PINNs satisfy governing laws of physics in the loss function without any training data. In this work, we solve various incompressible thermal convection problems, and compare the results with numerical or analytical results. To evaluate the accuracy of the model we solve a channel problem with an analytical solution. The model is highly dependent on the weights of individual loss terms. Increasing the weight of boundary condition loss improves the accuracy if the flow inside the domain is not complicated. To assess the performance of different type of networks and ability to capture the Neumann boundary conditions, we solve a thermal convection problem in a closed enclosure in which the flow occurs due to the temperature gradients on the boundaries. The simple fully connected network performs well in thermal convection problems, and we do not need a Fourier mapping in the network since there is no multiscale behavior. Lastly, we consider steady and unsteady partially blocked channel problems resembling industrial applications to power electronics and show that the method can be applied to transient problems as well.

# Citation

	@article{aygun_physics_2022,
		title = {Physics informed neural networks for two dimensional incompressible thermal convection problems},
		issn = {1300-3615},
		doi = {10.47480/isibted.1194992},
		urldate = {2022-11-15},
		journal = {Isı Bilimi ve Tekniği Dergisi},
		author = {Aygun, Atakan and Karakus, Ali},
		month = oct,
		year = {2022},
		pages = {221--232},
	}
