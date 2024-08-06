import numpy as np
from numpy import matlib

#INTERSECTIONS Intersections of curves.
#   Computes the (x,y) locations where two curves intersect.  The curves
#   can be broken with NaNs or have vertical segments.
#
# Example:
#   [X0,Y0] = intersections(X1,Y1,X2,Y2,ROBUST);
#
# where X1 and Y1 are equal-length vectors of at least two points and
# represent curve 1.  Similarly, X2 and Y2 represent curve 2.
# X0 and Y0 are column vectors containing the points at which the two
# curves intersect.
#
# ROBUST (optional) set to 1 or true means to use a slight variation of the
# algorithm that might return duplicates of some intersection points, and
# then remove those duplicates.  The default is true, but since the
# algorithm is slightly slower you can set it to false if you know that
# your curves don't intersect at any segment boundaries.  Also, the robust
# version properly handles parallel and overlapping segments.
#
# The algorithm can return two additional vectors that indicate which
# segment pairs contain intersections and where they are:
#
#   [X0,Y0,I,J] = intersections(X1,Y1,X2,Y2,ROBUST);
#
# For each element of the vector I, I(k) = (segment number of (X1,Y1)) +
# (how far along this segment the intersection is).  For example, if I(k) =
# 45.25 then the intersection lies a quarter of the way between the line
# segment connecting (X1(45),Y1(45)) and (X1(46),Y1(46)).  Similarly for
# the vector J and the segments in (X2,Y2).
#
# You can also get intersections of a curve with itself.  Simply pass in
# only one curve, i.e.,
#
#   [X0,Y0] = intersections(X1,Y1,ROBUST);
#
# where, as before, ROBUST is optional.

# Version: 1.12, 27 January 2010
# Author:  Douglas M. Schwarz
# Email:   dmschwarz=ieee*org, dmschwarz=urgrad*rochester*edu
# Real_email = regexprep(Email,{'=','*'},{'@','.'})


# Theory of operation:
#
# Given two line segments, L1 and L2,
#
#   L1 endpoints:  (x1(1),y1(1)) and (x1(2),y1(2))
#   L2 endpoints:  (x2(1),y2(1)) and (x2(2),y2(2))
#
# we can write four equations with four unknowns and then solve them.  The
# four unknowns are t1, t2, x0 and y0, where (x0,y0) is the intersection of
# L1 and L2, t1 is the distance from the starting point of L1 to the
# intersection relative to the length of L1 and t2 is the distance from the
# starting point of L2 to the intersection relative to the length of L2.
#
# So, the four equations are
#
#    (x1(2) - x1(1))*t1 = x0 - x1(1)
#    (x2(2) - x2(1))*t2 = x0 - x2(1)
#    (y1(2) - y1(1))*t1 = y0 - y1(1)
#    (y2(2) - y2(1))*t2 = y0 - y2(1)
#
# Rearranging and writing in matrix form,
#
#  [x1(2)-x1(1)       0       -1   0;      [t1;      [-x1(1);
#        0       x2(2)-x2(1)  -1   0;   *   t2;   =   -x2(1);
#   y1(2)-y1(1)       0        0  -1;       x0;       -y1(1);
#        0       y2(2)-y2(1)   0  -1]       y0]       -y2(1)]
#
# Let's call that A*T = B.  We can solve for T with T = A\B.
#
# Once we have our solution we just have to look at t1 and t2 to determine
# whether L1 and L2 intersect.  If 0 <= t1 < 1 and 0 <= t2 < 1 then the two
# line segments cross and we can include (x0,y0) in the output.
#
# In principle, we have to perform this computation on every pair of line
# segments in the input data.  This can be quite a large number of pairs so
# we will reduce it by doing a simple preliminary check to eliminate line
# segment pairs that could not possibly cross.  The check is to look at the
# smallest enclosing rectangles (with sides parallel to the axes) for each
# line segment pair and see if they overlap.  If they do then we have to
# compute t1 and t2 (via the A\B computation) to see if the line segments
# cross, but if they don't then the line segments cannot cross.  In a
# typical application, this technique will eliminate most of the potential
# line segment pairs.

def find_intersection_list(x1, y1, x2, y2):
    # x1 and y1 must be vectors with same number of points (at least 2).
    len_x1, len_y1, len_x2, len_y2 = len(x1), len(y1), len(x2), len(y2)
    if (len_x1 <2) or (len_y1 < 2) or len_x1 != len_y1:
        print('X1 and Y1 must be equal-length vectors of at least 2 points.')
    # x2 and y2 must be vectors with same number of points (at least 2).
    if (len_x2 <2) or (len_y2 < 2) or len_x2 != len_y2:
        print('X2 and Y2 must be equal-length vectors of at least 2 points.')
    
    return find_intersection(np.array([x1, y1]), np.array([x2, y2]))

def find_self_intersection_list(x, y):
    # x and y must be vectors with same number of points (at least 2).
    len_x, len_y = len(x), len(y)
    if (len_x <2) or (len_y < 2) or len_x != len_y:
        print('X and Y must be equal-length vectors of at least 2 points.')
    return find_self_intersection(np.array([x,y]))

def find_self_intersection(xy):
    return find_intersection(xy, xy, True)

def find_zero_intersections(x, y):
    (cz,) = np.nonzero((np.minimum(y[:-1], y[1:]) < 0) & (np.maximum(y[:-1], y[1:]) > 0))
    dx, dy = np.diff(x), np.diff(y)
    du = np.abs(y[cz]) / (np.abs(y[cz]) + np.abs(y[cz + 1]))
    return (x[cz] + dx[cz] * du), (y[cz] + dy[cz] * du)

def find_intersection(xy1, xy2, self_intersect = False):
    # Compute number of line segments in each curve and some differences we'll
    # need later.
    n1 = xy1.shape[1]-1
    n2 = xy2.shape[1]-1
    dxy1 = np.diff(xy1)
    dxy2 = np.diff(xy2)

    # Determine the combinations of i and j where the rectangle enclosing the
    # i'th line segment of curve 1 overlaps with the rectangle enclosing the
    # j'th line segment of curve 2.
    [j, i] = np.nonzero((matlib.repmat(np.minimum(xy1[0, 0:-1], xy1[0, 1:]), n2, 1) <= matlib.repmat(np.maximum(xy2[0, 0:-1], xy2[0, 1:]), n1, 1).T) &
        (matlib.repmat(np.maximum(xy1[0, 0:-1], xy1[0, 1:]), n2, 1) >= matlib.repmat(np.minimum(xy2[0, 0:-1], xy2[0, 1:]), n1, 1).T) &
        (matlib.repmat(np.minimum(xy1[1, 0:-1], xy1[1, 1:]), n2, 1) <= matlib.repmat(np.maximum(xy2[1, 0:-1], xy2[1, 1:]), n1, 1).T) &
        (matlib.repmat(np.maximum(xy1[1, 0:-1], xy1[1, 1:]), n2, 1) >= matlib.repmat(np.minimum(xy2[1, 0:-1], xy2[1, 1:]), n1, 1).T))


    # Find segments pairs which have at least one vertex = NaN and remove them.
    # This line is a fast way of finding such segment pairs.  We take
    # advantage of the fact that NaNs propagate through calculations, in
    # particular subtraction (in the calculation of dxy1 and dxy2, which we
    # need anyway) and addition.
    # At the same time we can remove redundant combinations of i and j in the
    # case of finding intersections of a line with itself.
    if self_intersect:
        remove = np.isnan(sum(dxy1[:, i] + dxy2[:, j], 2)) | j <= i + 1
    else:
        remove = np.isnan(sum(dxy1[:, i] + dxy2[:, j], 2))

    i[remove] = []
    j[remove] = []

    # Initialize matrices.  We'll put the t's and b's in matrices and use them
    # one column at a time.  aa is a 3-D extension of A where we'll use one
    # plane at a time.
    n = len(i)
    t = np.zeros([4,n])
    aa = np.zeros([4,4,n])
    aa[[0, 1], 2, :] = -1
    aa[[2, 3], 3, :] = -1
    aa[[0, 2], 0, :] = dxy1[:, i]
    aa[[1, 3], 1, :] = dxy2[:, j]
    b = -1*np.array([xy1[0, i], xy2[0, j], xy1[1, i], xy2[1, j]])

    # Loop through possibilities.  Trap singularity warning and then use
    # lastwarn to see if that plane of aa is near singular.  Process any such
    # segment pairs to determine if they are colinear (overlap) or merely
    # parallel.  That test consists of checking to see if one of the endpoints
    # of the curve 2 segment lies on the curve 1 segment.  This is done by
    # checking the cross product
    #
    #   (x1(2),y1(2)) - (x1(1),y1(1)) x (x2(2),y2(2)) - (x1(1),y1(1)).
    #
    # If this is close to zero then the segments overlap.

    # If the robust option is false then we assume no two segment pairs are
    # parallel and just go ahead and do the computation.  If A is ever singular
    # a warning will appear.  This is faster and obviously you should use it
    # only when you know you will never have overlapping or parallel segment
    # pairs.

    overlap = np.zeros(n)

    # Use try-catch to guarantee original warning state is restored.
    for k in range(n):
        t[:,k] = np.linalg.lstsq(aa[:,:,k], b[:, k], rcond=None)[0]
        

    # Find where t1 and t2 are between 0 and 1 and return the corresponding
    # x0 and y0 values.
    in_range = np.array(((t[0, :] >= 0) & (t[1, :] >= 0) & (t[0, :] <= 1) & (t[1, :] <= 1))).T
    # For overlapping segment pairs the algorithm will return an
    # intersection point that is at the center of the overlapping region.
    if any(overlap):
        ia = i[overlap]
        ja = j[overlap]
    	# set x0 and y0 to middle of overlapping region.
        t[2, overlap] = (np.maximum(np.minimum(xy1[0, ia], xy1[0, ia+1]), np.minimum(xy2[0, ja], xy2[0, ja+1])) + np.minimum(np.maximum(xy1[0, ia], xy1[0, ia+1]), np.maximum(xy2[0, ja], xy2[0, ja+1])))/2
        t[3, overlap] = (np.maximum(np.minimum(xy1[1, ia], xy1[1, ia+1]), np.minimum(xy2[1, ja], xy2[1, ja+1])) + np.minimum(np.maximum(xy1[1, ia], xy1[1, ia+1]), np.maximum(xy2[1, ja], xy2[1, ja+1])))/2
        selected = in_range or overlap
    else:
        selected = in_range

    # Remove duplicate intersection points.
    [xy0, index] = np.unique(t[2:4, selected].T, return_index=True, axis=0)
    x0 = xy0[:, 0]
    y0 = xy0[:, 1]

    sel_index = np.argwhere([selected])
    sel = sel_index[index]
    iout = i[sel] + t[0, sel]
    jout = j[sel] + t[1, sel]
    return x0, y0, iout, jout

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # run = np.arange(20)
    a1 = np.array([1, 2, 3, 4, 5])
    a2 = np.array([1, 2, 3, 4, 5])
    a3 = np.array([1.5, 2.5, 3.5, 4.5])
    a4 = np.array([4, 3, 2, 1])
    a5 = np.linspace(0, 10, 20)
    a6 = a5+np.sin(a5)
    line1 = np.array([a1, a2])
    line2 = np.array([a3, a4])
    line3 = np.array([a5, a5])
    line4 = np.array([a5, a6])
    x0, y0, iout, jout = find_intersection(line3, line4)

    plt.plot(a5, a5)
    plt.plot(a5, a6)
    plt.scatter(x0, y0)
    # plt.scatter(iout, jout)
    plt.show()
