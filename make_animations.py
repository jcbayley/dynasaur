import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def make_2d_animation(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): _description_
        masses (_type_): _description_
        true_timeseries (_type_): _description_
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = len(masses)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.set_xlim([np.min(timeseries[:,0,:]),np.max(timeseries[:,0,:])])
    ax.set_ylim([np.min(timeseries[:,1,:]),np.max(timeseries[:,1,:])])


    # Create particles as lines
    particles = [ax.plot(0, 0, marker="o", markersize=masses[mind]*10) for mind in range(num_masses)]

    true_particles = [ax.plot(0, 0, marker="o", markersize=true_masses[mind]*10, color="k") for mind in range(num_masses)]

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            x, y = timeseries[mind][:, frame]
            particles[mind][0].set_data(x, y)

            xt, yt = true_timeseries[mind][:, frame]
            true_particles[mind][0].set_data(xt, yt)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"animation_{index}.gif"), writer=writergif)

def make_2d_distribution(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): _description_
        masses (_type_): _description_
        true_timeseries (_type_): _description_
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = len(masses[0])

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.set_xlim([np.min(timeseries[:,0,:]),np.max(timeseries[:,0,:])])
    ax.set_ylim([np.min(timeseries[:,1,:]),np.max(timeseries[:,1,:])])


    # Create particles as lines
    #particles = [ax.plot(timeseries[:,mind,0,0], timeseries[:,mind,1,0], marker="o", ls="none",markersize=masses[0,mind]*10) for mind in range(num_masses)]
    particles = [ax.scatter(timeseries[:,mind,0,0], timeseries[:,mind,1,0],s=masses[:,mind]*10) for mind in range(num_masses)]

    true_particles = [ax.plot(true_timeseries[0,0,:], true_timeseries[0,1,:], marker="o", markersize=true_masses[mind]*10, color="k") for mind in range(num_masses)]

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            #print(np.shape(timeseries[:,mind][:,:,frame]))
            x, y = timeseries[:,mind][:,:,frame].reshape(2, len(timeseries))
            #particles[mind][0].set_data(x, y)
            particles[mind].set_offsets(np.c_[x, y])

            xt, yt = true_timeseries[mind][:,frame]
            true_particles[mind][0].set_data(xt, yt)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"multi_animation_{index}.gif"), writer=writergif)


def make_3d_animation(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): [masses, dimension, timestep]
        masses (_type_): _description_
        true_timeseries (_type_): [masses, dimension, timestep]
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = np.shape(masses)[-1]

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([np.min(timeseries[:,0,:]),np.max(timeseries[:,0,:])])
    ax.set_ylim([np.min(timeseries[:,1,:]),np.max(timeseries[:,1,:])])
    ax.set_zlim([np.min(timeseries[:,2,:]),np.max(timeseries[:,2,:])])


    # Create particles as lines
    particles = ax.scatter(timeseries[:,0,0], timeseries[:,1,0], timeseries[:,2,0], s=masses*10)

    true_particles = ax.scatter(true_timeseries[:,0,0], true_timeseries[:,1,0], true_timeseries[:,2,0], s=true_masses*10) 

    #true_particles = [ax.scatter(0, 0, 0, s=true_masses[mind]*10, color="k") for mind in range(num_masses)]

    def update_plot(frame):
        # Set new positions for each particle based on the current frame
        x, y, z = np.transpose(timeseries[:,:,frame], (1,0))
        particles._offsets3d = (x, y, z)
        #particles[mind][0].set_data(x, y)
        #particles[mind][0].set_3d_properties(z)

        #print(np.shape(true_timeseries), np.shape(true_timeseries[mind]))
        xt, yt, zt = np.transpose(true_timeseries[:,:,frame], (1,0))
        true_particles._offsets3d = (xt, yt, zt)
        #true_particles[mind][0].set_data(xt, yt)
        #true_particles[mind][0].set_3d_properties(zt)
        #print("update")


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1, blit=False)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"animation_{index}.gif"), writer=writergif)

def make_3d_distribution(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): shape [sample, masses, dimension, timestep]
        masses (_type_): _description_
        true_timeseries (_type_): shape [sample, masses, dimension, timestep]
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = np.shape(masses)[-1]

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim([np.min(timeseries[:,:,0,:]),np.max(timeseries[:,:,0,:])])
    ax.set_ylim([np.min(timeseries[:,:,1,:]),np.max(timeseries[:,:,1,:])])
    ax.set_ylim([np.min(timeseries[:,:,2,:]),np.max(timeseries[:,:,2,:])])


    # Create particles as lines
    #particles = [ax.plot(timeseries[:,mind,0,0], timeseries[:,mind,1,0], timeseries[:,mind,2,0], marker="o", ls="none",markersize=masses[0,mind]*10) for mind in range(num_masses)]
    particles = [ax.scatter(timeseries[:,mind,0,0], timeseries[:,mind,1,0],timeseries[:,mind,2,0],s=masses[:,mind]*10) for mind in range(num_masses)]

    #true_particles = [ax.plot(true_timeseries[mind,0,0], true_timeseries[mind,1,0], true_timeseries[mind,2,0], marker="o", ls="none", color="k", markersize=true_masses[mind]*10) for mind in range(num_masses)]
    true_particles = ax.scatter(true_timeseries[:,0,0], true_timeseries[:,1,0], true_timeseries[:,2,0],s=true_masses*10, color="k")

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            #print(np.shape(timeseries[:,mind][:,:,frame]))
            x, y, z = np.transpose(timeseries[:,mind,:,frame], (1,0))
            #particles[mind][0].set_data(x, y)
            #particles[mind][0].set_3d_properties(z)
            particles[mind]._offsets3d = (x, y, z)

        xt, yt, zt = np.transpose(true_timeseries[:,:,frame], (1,0))
        #true_particles[mind][0].set_data(xt, yt)
        #true_particles[mind][0].set_3d_properties(zt)
        true_particles._offsets3d = (xt, yt, zt)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"multi_animation_{index}.gif"), writer=writergif)
