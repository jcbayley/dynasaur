# dynasaur

Project to find the mass and dynamics of a system from the gravitational wavestrain time series. 
Use normalising flows conditioned on strain to find polynomial coefficients of mass motion

For a test case we trained a model on circular orbits with random radii, masses and initial phase angle. These were simulated in the x,y plane, however were reconstructed in 3d space.

The black points in the left panel are the true injected point masses where their true waveform is shown in the right hand panel in black.
The left panel of each of these plots also shows many pairs of	blue and orange	points,	these are points sampled from a	normalising flow conditioned on	the black waveforms in	the left panels. For each of these pairs we can	reconstruct the	waveform again,	the 90%	confidence region of all of these reconstructions are shown in	the right hand plots.

![Alt Text](https://media.githubusercontent.com/media/jcbayley/dynasaur/main/figures/animation_circularbinary_2.gif)

The main goal here was to train on lots of random orbits, where the fourier components of the positions of the masses are randomly sampled between [0,1]. 

![Alt Text](https://media.githubusercontent.com/media/jcbayley/dynasaur/main/figures/animation_random_0.gif)

# Installation

In the root directory run:

```bash
pip install .
```

# Usage

To use train a model based on a config (examples can be found in the examples directory):

```bash
python -m dynasaur.train_model --config config.ini --train
```

To test the model 

```bash
python -m dynasaur.train_model --config config.ini --test --ntest 10 --makeplots
```
