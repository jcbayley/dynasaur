import rebound


def solve_ode():

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.add(m=0.5)
    sim.add(m=0.5)

    sim.move_to_com()

    sim.integrate(1)

    for o in sim.orbits():
        print(o)

if __name__ == "__main__":
    solve_ode()