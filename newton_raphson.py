import numpy as np
import matplotlib.pyplot as mp

# Equiation : f(U) + H(U) * V = 0
# <=> H(U) * V = -f(U)

# Transforms an array of functions to a function returning an array of the values of each function
def function_array(f):
	return lambda U: np.vectorize(lambda fn: fn(U))(f)

# Produces a value indicating how x is near to a root of f. Returns a value close to 0 if x is near a root and a value far from 0 if x is far from any root.
def phi(f, x):
	fx = f(x)
	return np.matmul(fx.T, fx)

# Finds a root of f where J is the Jacobian matrix of j.
# Parameters:
# - U0 the starting point of the algorithm
# - N the maximum number if iterations
# - epsilon the desired accuracy
# - backtracking wether backtraking should be used
# - track_phi wether the value of phi(Un) should be tracked and returned in an array
# Returns:
# - A root of f if track_phi is False
# - A root of f and the array of phi(Un) if track_phi is True
def Newton_Raphson(f, J, U0, N, epsilon, backtracking = True, track_phi = False):

	Un = U0
	previousPhi = phi(f, U0)
	if track_phi:
		phiArray = [previousPhi]

	for i in range(N):
		fUn = f(Un)
		if np.linalg.norm(fUn) < epsilon:
			break;

		V = np.linalg.lstsq(J(Un), -fUn, rcond=-1)[0]

		# Backtracking
		step = 1.0
		newUn = Un + V
		newPhi = phi(f, newUn)
		if backtracking:
			while newPhi >= previousPhi:
				step *= 0.9;
				newUn = Un + step * V
				newPhi = phi(f, newUn)

		Un = newUn
		previousPhi = newPhi
		if track_phi:
			phiArray.append(newPhi)

	if track_phi:
		return (Un, phiArray)
	else:
		return Un

# Tests the Newton_Raphson algorithm with simple cases
def test_Newton_Raphson():
	# f1(x) = x² - 2x + 1
	# f1'(x) = 2x - 2
	f1 = function_array([lambda x: x**2 - 2*x + 1])
	J1 = function_array([[lambda x: 2*x - 2]])
	expected = 1
	root1 = Newton_Raphson(f1, J1, np.array([0]), 40, 10e-6)
	print("f(x) = x² - 2x + 1")
	print("Root:", root1)
	print("Relative error:", np.abs(root1 - expected))


	# f2(x, y) = (x² + y, x + 3y + 4)
	# df2/dx(x, y) = (2*x, 1)
	# df2/dy(x, y) = (1, 3)
	f2 = function_array([lambda z: z[0]**2 + z[1], lambda z: z[0] + 3*z[1] + 4])
	J2 = function_array([[lambda z: 2*z[0], lambda z: 1],
						 [lambda z: 1, lambda z: 3]])
	root2 = Newton_Raphson(f2, J2, np.array([1, 4]), 40, 1e-10)
	print("f(x, y) = (x² + y, x + 3y + 4)")
	print("Root:",root2)
	print("f(root) =", f2(root2))

# Tests the effets of backtracking on the convergence speed
def test_backtracking():
	# f(x) = x^3 - 2x^2 + 1
	# f'(x) = 3x^2 - 4x
	f = function_array([lambda x: x**3 - 2*x**2 + 1])
	J = function_array([[lambda x: 3*(x**2) - 4*x]])
	root0, phiArray0 = Newton_Raphson(f, J, np.array([0.01]), 40, 10e-10, False, True)
	root1, phiArray1 = Newton_Raphson(f, J, np.array([0.01]), 40, 10e-10, True, True)


	mp.plot(range(len(phiArray0)), phiArray0, label = "Without Backtracking", linewidth = 1.0)
	mp.plot(range(len(phiArray1)), phiArray1, label = "With Backtracking", linewidth = 1.0)
	mp.yscale('log')
	mp.legend()
	mp.title("The value of $phi(U_n)$ in each iteration, with $f(x) = x^3 - 2x^2 + 1$, $U_0 = 0.01$ and $\\epsilon = 10^{-10}$")
	mp.xlabel("n, the number of iterations")
	mp.ylabel("$phi(U_n)$")
	mp.show()


if __name__ == "__main__":
	test_Newton_Raphson()
	test_backtracking()