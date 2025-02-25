# from numba import njit
import numpy as np

# @njit
def skew(x):
    # np.dot(skew(x), y) = np.cross(x, y)
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
# @njit
def cross_2d(u, v):
    return u[0] * v[1] - u[1] * v[0]

# @njit
def norm_2d(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2)


# @njit
def is_point_in_triangle(p, p1, p2, p3):
    # Project points onto the plane defined by p1, p2, and p3
    # we need to make the points 2d so we can use signed distances
    # Following this stack overflow answer: https://stackoverflow.com/questions/74620278/project-3d-points-to-2d-points-in-python
    plane_normal = np.dot(skew(p2 - p1), p3 - p1)
    plane_normal /= np.linalg.norm(plane_normal)
    # Make a projection matrix
    y_vec = p2 - p1
    y_vec = y_vec / np.linalg.norm(y_vec)
    x_vec = np.dot(skew(plane_normal), y_vec)
    x_vec = x_vec / np.linalg.norm(x_vec)
    proj_mat = np.vstack((x_vec, y_vec))
    # Project the points
    projected_p1 = p1 @ proj_mat.T
    projected_p2 = p2 @ proj_mat.T
    projected_p3 = p3 @ proj_mat.T
    projected_p = p @ proj_mat.T
    
    # Check if the projected point is inside the triangle
    # This is done by checking if the area of the triangle is equal to the sum of the areas of the sub triangles
    area_total = cross_2d(projected_p2 - projected_p1, projected_p3 - projected_p1) / 2.0
    area_sub1 = cross_2d(projected_p2 - projected_p, projected_p3 - projected_p) / 2.0
    area_sub2 = cross_2d(projected_p1 - projected_p, projected_p3 - projected_p) / 2.0
    area_sub3 = cross_2d(projected_p1 - projected_p, projected_p2 - projected_p) / 2.0
    distances = np.zeros(3)
    if np.isclose(area_total, area_sub1 + area_sub2 + area_sub3, rtol=1e-10):
        return distances[0] # needs to be a float64 for numba
    
    distances[0] = cross_2d(projected_p2 - projected_p1, projected_p - projected_p1) / norm_2d(projected_p2 - projected_p1)
    distances[1] = cross_2d(projected_p3 - projected_p2, projected_p - projected_p2) / norm_2d(projected_p3 - projected_p2)
    distances[2] = cross_2d(projected_p1 - projected_p3, projected_p - projected_p3) / norm_2d(projected_p1 - projected_p3)
    
    sign_of_inward = np.sign(area_total / norm_2d(projected_p2 - projected_p1))
    if sign_of_inward > 0:
        distances *= -1.0
    return distances.max()

# @njit
def db_for_point_in_triangle(p, p1, p2, p3):
    # Project points onto the plane defined by p1, p2, and p3
    # we need to make the points 2d so we can use signed distances
    # Following this stack overflow answer: https://stackoverflow.com/questions/74620278/project-3d-points-to-2d-points-in-python
    plane_normal = np.dot(skew(p2[:3] - p1[:3]), p3[:3] - p1[:3])
    plane_normal /= np.linalg.norm(plane_normal)
    # Make a projection matrix
    y_vec = p2[:3] - p1[:3]
    y_vec = y_vec / np.linalg.norm(y_vec)
    x_vec = np.dot(skew(plane_normal), y_vec)
    x_vec = x_vec / np.linalg.norm(x_vec)
    proj_mat = np.vstack((x_vec, y_vec))
    # Project the points
    projected_p1 = p1[:3] @ proj_mat.T
    projected_p2 = p2[:3] @ proj_mat.T
    projected_p3 = p3[:3] @ proj_mat.T
    projected_p = p[:3] @ proj_mat.T
    # Compute the barycentric coordinates of projected_p
    detT = (projected_p2[1] - projected_p3[1]) * (projected_p1[0] - projected_p3[0]) + (projected_p3[0] - projected_p2[0]) * (projected_p1[1] - projected_p3[1])
    w1 = ((projected_p2[1] - projected_p3[1]) * (projected_p[0] - projected_p3[0]) + (projected_p3[0] - projected_p2[0]) * (projected_p[1] - projected_p3[1])) / detT
    w2 = ((projected_p3[1] - projected_p1[1]) * (projected_p[0] - projected_p3[0]) + (projected_p1[0] - projected_p3[0]) * (projected_p[1] - projected_p3[1])) / detT
    w3 = 1.0 - w1 - w2

    # Interpolate the delta and beta values
    interpolated_delta = w1 * p1[3] + w2 * p2[3] + w3 * p3[3]
    interpolated_beta = w1 * p1[4] + w2 * p2[4] + w3 * p3[4]

    return interpolated_delta, interpolated_beta

# the intention of this function is to interp with linear interpolation beyond the bounds of the array
def clean_interp(x, xp, fp):
    ret_val = np.interp(x, xp, fp)
    lower_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    upper_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    ret_val[x < xp[0]] = fp[0] + lower_slope * (x[x < xp[0]] - xp[0])
    ret_val[x > xp[-1]] = fp[-1] + upper_slope * (x[x > xp[-1]] - xp[-1])
    return ret_val
    

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)

def to_vel_frame(x, y, beta):
    return x * np.cos(beta) - y * np.sin(beta), x * np.sin(beta) + y * np.cos(beta)

def to_car_frame(x, y, beta):
    return x * np.cos(beta) + y * np.sin(beta), -x * np.sin(beta) + y * np.cos(beta)

def vel_at_tire(v, omega, beta, x, y):
    v_x = v * np.cos(beta) - omega * y
    v_y = v * np.sin(beta) - omega * x
    v_v = np.sqrt(v_x**2 + v_y**2)
    return v_v

UP = np.deg2rad(15)
def clip(x, up=UP):
    return min(max(x, -UP), UP)

def calculate_curvature(points):
    """
    Calculate the curvature of a line represented by a [2,n] numpy array of points.

    Parameters:
    points (numpy.ndarray): A [2, n] array where the first row contains the x-coordinates
                            and the second row contains the y-coordinates of the points.

    Returns:
    numpy.ndarray: An array of curvature values at each point.
    """

    # Ensure the input is a numpy array
    points = np.asarray(points)
    
    # First derivatives
    dx_dt = np.gradient(points[0])
    dy_dt = np.gradient(points[1])

    # Second derivatives
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Curvature formula
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5

    return curvature

def interpolate(inputs, outputs, target):
    # Ensure inputs and outputs are the same length
    if len(outputs) != len(inputs):
        raise ValueError("The length of outputs and inputs must be the same.")

    lower_slope = (outputs[1] - outputs[0]) / (inputs[1] - inputs[0])
    upper_slope = (outputs[-1] - outputs[-2]) / (inputs[-1] - inputs[-2])

    scale = 100 #really dump way to account for out of bounds stuff but I'm not good enough at python and clean_interp method won't work idk why
    #inputs.insert(0, lower_slope * scale + inputs[0])
    #inputs.append(upper_slope * scale + inputs[len(inputs) - 1])
    #outputs.insert(0, lower_slope * scale + outputs[0])
    #outputs.append(upper_slope * scale + outputs[len(outputs) - 1])


    if hasattr(target, "__len__"):
        targets = []
        for val in target:

            # Loop through the outputs array to find the segment
            for i in range(len(inputs) - 1):
                if inputs[i] <= val <= inputs[i + 1] or inputs[i] >= val >= inputs[i + 1]:
                    # Perform linear interpolation
                    x0, x1 = outputs[i], outputs[i + 1]
                    y0, y1 = inputs[i], inputs[i + 1]
                    interpolated_output = x0 + (val - y0) * (x1 - x0) / (y1 - y0)
                    targets.append(interpolated_output) 

            if val < inputs[0]:
                targets.append(outputs[0] + ( lower_slope * ( val - inputs[0] ) ))
            elif val > inputs[len(inputs) - 1]:
                targets.append(outputs[len(outputs) - 1] + ( upper_slope * ( val - inputs[len(inputs) - 1] ) ))

        return targets

    else:

        for i in range(len(inputs) - 1):
            if inputs[i] <= target <= inputs[i + 1] or inputs[i] >= target >= inputs[i + 1]:
                # Perform linear interpolation
                x0, x1 = outputs[i], outputs[i + 1]
                y0, y1 = inputs[i], inputs[i + 1]
                interpolated_output = x0 + (target - y0) * (x1 - x0) / (y1 - y0)

                return interpolated_output

        if target < inputs[0]:
            return outputs[0] + ( lower_slope * ( target - inputs[0] ) )
        elif target > inputs[len(inputs) - 1]:
            return outputs[len(outputs) - 1] + ( upper_slope * ( target - inputs[len(inputs) - 1] ) )

    # If target is not in the range of inputs, raise an error
    raise ValueError("Target outputs is outside the range of provided inputs.")

# slip angle control
def sa_lut(v):
    velocities = [5, 6, 7, 8, 10, 13, 15, 20, 25, 30]
    max_slip = [35, 31, 28, 24, 18, 14.25, 13.25, 12.75, 14, 14]

    velocities = np.array(velocities)
    max_slip = np.array(max_slip)

    return interpolate(velocities, max_slip, v)