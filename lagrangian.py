import numpy as np
import newton_raphson as nr

# Creates a gravitational force function
def gravitational_force(k, z0):
    return lambda z: -k * (z - z0) / (((z-z0)**2).sum()**(3/2))

# Creates the jacobian matix correspounding to a gravitational force
def gravitional_jacob(k, z0):
    def res(z):
        v = ((z[0] - z0[0])**2 + (z[1] - z[1])**2 )**(3/2)
        x = z[0]
        y = z[1]
        x0 = z0[0]
        y0 = z0[1]
        resx_x = -k * (v - (x - x0) *(3/2) * ((x-x0)**2 + (y - y0)**2)**(1/2) * 2 * (x - x0))/ (v ** 2)
        resx_y = -k * ( - (x - x0) *(3/2) * ((x-x0)**2 + (y - y0)**2)**(1/2) * 2 * (y - y0 ))/ (v ** 2)
        resy_x = -k * ( - (y - y0) *(3/2) * ((x-x0)**2 + (y - y0)**2)**(1/2) * 2 * (x - x0 ))/ (v ** 2)
        resy_y = -k * (v - (y - y0) *(3/2) * ((x-x0)**2 + (y - y0)**2)**(1/2) * 2 * (y - y0))/ (v ** 2)        
        return np.matrix([[resx_x,resx_y],[resy_x, resy_y]])
    return res;

# Creates a centrifigual force function
def centrifigual_force(k, z0):
    return lambda z : k * (z - z0)

# Creates the jacobian matix correspounding to a centrifigual force
def centrifigual_jacob(k, z0):
    return lambda z : np.matrix([[k,0],[0,k]])

# Creates an elastic force function
def elastic_force(k, z0):
    return lambda z: - k * (z - z0)

# Creates the jacobian matix correspounding to an elastic force
def elastic_jacob(k, z0):
    return lambda z : np.matrix([[-k,0],[0,-k]])

# Apply two gravitational forces to z
def forces1(z):
    return gravitational_force(1,np.array([0,0]))(z) + gravitational_force(0.01, np.array([1,0]))(z)

# Apply two gravitational forces and a centrifigual force to z
def forces2(z):
    return forces1(z) + centrifigual_force(1, np.array([0.01/1.01, 0]))(z)

# Produces the jacobian matix of two gravitational forces aplied to z
def jacobians1(z):
    return gravitional_jacob(1,np.array([0,0]))(z) + gravitional_jacob(0.01, np.array([1,0]))(z)

# Produces the jacobian matix of two gravitational forces and a centrifigual force aplied to z
def jacobians2(z):
    return jacobians1(z) + centrifigual_jacob(1, np.array([0.01/1.01, 0]))(z)

# Finds an equilibrium point for forces1 and forces2
def solve_two_grav():
    print("First case : Two gravitational forces with respective coefficients 1 (resp. 0.01) originating from [0,0] (resp. [1,0])")
    Eq1 = nr.Newton_Raphson(forces1, jacobians1, np.array([1.5,0]), 1000, 1e-10);
    print("Equilibrium points =", Eq1)
    print("f(Eq) =", forces1(Eq1))

    print("Second case : previous forces and a centrifugal force centered on the barycenter of the two masses, with coefficient 1")
    Eq2 = nr.Newton_Raphson(forces2, jacobians2, np.array([1.5,0]), 1000, 1e-10);
    print("Equilibrium points =", Eq2)
    print("f(Eq) =", forces2(Eq2))

# Tests forces using an example
def test_forces():
    U = np.array([1.5,0])
    print("U =", U)
    print("f(U) =", forces2(U))
    print("df(U) =\n", jacobians2(U))

# Finds an equilibrium point for forces2 depending on U0
def find_lagragian_points(U0):
    # U0 = (0.5,  0)  -> U = ( 0.85927766, 0 ) = L1 
    # U0 = (1.5,  0)  -> U = ( 1.15775715, 0 ) = L2 
    # U0 = (-1.5, 0)  -> U = (-0.99754112, 0 ) = L3 
    
    U = nr.Newton_Raphson(forces2, jacobians2, U0, 1000, 1e-10)
    print("U =", U)
    print("f(U) =", forces2(U))
    print("df(U) =\n", jacobians2(U))

if __name__ == "__main__":
    test_forces()
    solve_two_grav()
