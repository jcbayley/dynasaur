# massdynamics

Project to find the mass dynamics of a system from the strain time series. 
Use normalising flows conditioned on strain to find polynomial coefficients of mass motion

For a test case we trained a model on circular orbits with random radii, masses and initial phase angle. These were simulated in the x,y plane, however were reconstructed in 3d space.

![Alt Text](https://media.githubusercontent.com/media/jcbayley/massdynamics/main/figures/animation_circularbinary_2.gif)

The main goal here was to train on lots of random orbits, where the fourier components of the positions of the masses are randomly sampled betwee [0,1].

![Alt Text](https://media.githubusercontent.com/media/jcbayley/massdynamics/main/figures/animation_random_0.gif)
