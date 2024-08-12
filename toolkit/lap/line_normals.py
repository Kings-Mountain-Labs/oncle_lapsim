import numpy as np

def linenormals2d(Vertices):
    """This function calculates the normals, of the line points
    using the neighbouring points of each contour point, and 
    forward an backward differences on the end points   
    n=Linenormals2d(V,L)    
    inputs,
      V : List of points/vertices 2 x M
    (optional)
      Lines : A n x 2 list of line pieces, by indices of the vertices
            (if not set assume Lines=[1 2; 2 3 ; ... ; M-1 M])  
    outputs,
      n : The normals of the Vertices 2 x M 
    Example, Hand
     load('testdata'); 
     n=Linenormals2d(Vertices,Lines);
     figure,
     plot([Vertices(:,1) Vertices(:,1)+10*n(:,1)]',[Vertices(:,2) Vertices(:,2)+10*n(:,2)]');   
    Function is written by d.Kroon University of Twente (August 2011)
    If no line-indices, assume a x(1) connected with x(2), x(3) with x(4) ..."""
    lines = np.array([np.arange(Vertices.shape[0]-1), np.arange(1, Vertices.shape[0])]).T


    # Calculate tangent vectors
    dt = Vertices[lines[:, 0], :] - Vertices[lines[:, 1], :]

    # Make influence of tangent vector 1/distance
    # (Weighted Central differences. Points which are closer give a 
    # more accurate estimate of the normal)
    ll = np.sqrt(dt[:, 0]**2 + dt[:, 1]**2)
    dt[:, 0] = dt[:, 0] / max(ll**2)
    dt[:, 1] = dt[:, 1] / max(ll**2)

    d1 = np.zeros(Vertices.shape)
    d1[lines[:, 0], :] = dt
    d2 = np.zeros(Vertices.shape)
    d2[lines[:, 1], :] = dt
    d = d1+d2

    # normalize the normal
    ll = np.sqrt(d[:, 0]**2 + d[:, 1]**2) + np.finfo(np.float64).eps # prevent a divide by zero error
    n = np.zeros([d.shape[0], 2])
    n[:, 0] = -d[:, 1] / ll
    n[:, 1] =  d[:, 0] / ll
    return n

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = np.arange(150)*0.1
    b = np.square(a)
    line = np.array([a, b])
    xyout = linenormals2d(line.T)
    print(xyout)
    plt.plot(a, b)
    plt.quiver(a, b, xyout[:, 0], xyout[:, 1])
    plt.show()