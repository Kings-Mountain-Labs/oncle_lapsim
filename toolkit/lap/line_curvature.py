import numpy as np

def linecurvature2D(vertices):
    """
    This function calculates the curvature of a 2D line. It first fits 
    polygons to the points. Then calculates the analytical curvature from
    the polygons;

     k = LineCurvature2D(Vertices,Lines)

    inputs,
      Vertices : A M x 2 list of line points.
      (optional)
      Lines : A N x 2 list of line pieces, by indices of the vertices
            (if not set assume Lines=[1 2; 2 3 ; ... ; M-1 M])

    outputs,
      k : M x 1 Curvature values



    Example, Circle
     r=sort(rand(15,1))*2*pi;
     Vertices=[sin(r) cos(r)]*10;
     Lines=[(1:size(Vertices,1))' (2:size(Vertices,1)+1)']; Lines(end,2)=1;
     k=LineCurvature2D(Vertices,Lines);

     figure,  hold on;
     N=LineNormals2D(Vertices,Lines);
     k=k*100;
     plot([Vertices(:,1) Vertices(:,1)+k.*N(:,1)]',[Vertices(:,2) Vertices(:,2)+k.*N(:,2)]','g');
     plot([Vertices(Lines(:,1),1) Vertices(Lines(:,2),1)]',[Vertices(Lines(:,1),2) Vertices(Lines(:,2),2)]','b');
     plot(sin(0:0.01:2*pi)*10,cos(0:0.01:2*pi)*10,'r.');
     axis equal;

    Example, Hand
     load('testdata');
     k=LineCurvature2D(Vertices,Lines);

     figure,  hold on;
     N=LineNormals2D(Vertices,Lines);
     k=k*100;
     plot([Vertices(:,1) Vertices(:,1)+k.*N(:,1)]',[Vertices(:,2) Vertices(:,2)+k.*N(:,2)]','g');
     plot([Vertices(Lines(:,1),1) Vertices(Lines(:,2),1)]',[Vertices(Lines(:,1),2) Vertices(Lines(:,2),2)]','b');
     plot(Vertices(:,1),Vertices(:,2),'r.');
     axis equal;

    Function is written by D.Kroon University of Twente (August 2011)
    """


    # Get left and right neighbor of each points
    lenz = vertices.shape[0]
    Na = np.zeros(lenz, dtype=int)
    Nb = np.zeros(lenz, dtype=int)
    Na[:-1] = np.arange(1, lenz)
    Na[-1] = lenz - 2
    Nb[0] = 1
    Nb[1:] = np.arange(lenz - 1)

    # Correct for sampling differences
    Ta = -np.sqrt(np.sum(np.square(vertices - vertices[Na, :]), axis=1))
    Tb =  np.sqrt(np.sum(np.square(vertices - vertices[Nb, :]), axis=1))

    # Fit a polygons to the vertices 
    # x=a(3)*t^2 + a(2)*t + a(1) 
    # y=b(3)*t^2 + b(2)*t + b(1) 
    # we know the x,y of every vertices and set t=0 for the vertices, and
    # t=Ta for left vertices, and t=Tb for right vertices,  
    x = np.array([vertices[Na, 0], vertices[:, 0], vertices[Nb, 0]]).T
    y = np.array([vertices[Na, 1], vertices[:, 1], vertices[Nb, 1]]).T

    invM = inverse3(Ta, Tb)
    a, b = np.zeros([invM.shape[0], 3]), np.zeros([invM.shape[0], 3])
    
    a[:, 0] = invM[:, 0, 0] * x[:, 0] + invM[:, 1, 0] * x[:, 1] + invM[:, 2, 0] * x[:, 2]
    a[:, 1] = invM[:, 0, 1] * x[:, 0] + invM[:, 1, 1] * x[:, 1] + invM[:, 2, 1] * x[:, 2]
    a[:, 2] = invM[:, 0, 2] * x[:, 0] + invM[:, 1, 2] * x[:, 1] + invM[:, 2, 2] * x[:, 2]
    b[:, 0] = invM[:, 0, 0] * y[:, 0] + invM[:, 1, 0] * y[:, 1] + invM[:, 2, 0] * y[:, 2]
    b[:, 1] = invM[:, 0, 1] * y[:, 0] + invM[:, 1, 1] * y[:, 1] + invM[:, 2, 1] * y[:, 2]
    b[:, 2] = invM[:, 0, 2] * y[:, 0] + invM[:, 1, 2] * y[:, 1] + invM[:, 2, 2] * y[:, 2]

    # Calculate the curvature from the fitted polygon
    return 2*(a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]) / ((a[:, 1]**2 + b[:, 1]**2 + np.finfo(np.float64).eps)**(3 / 2)) # prevent a divide by zero error

def  inverse3(Ta, Tb):
    # This function does inv(mat) , but then for an array of 3x3 matrices
    Ta_sq = np.square(Ta)
    Tb_sq = np.square(Tb)
    mat_s = Ta.shape[0]
    adjmat = np.zeros([mat_s, 3, 3])
    adjmat[:, 0, 0] =  np.zeros(mat_s)
    adjmat[:, 0, 1] = -Tb_sq
    adjmat[:, 0, 2] =  -Tb
    adjmat[:, 1, 0] =  Ta * Tb_sq - Tb * Ta_sq
    adjmat[:, 1, 1] =  Tb_sq - Ta_sq
    adjmat[:, 1, 2] =  Tb - Ta
    adjmat[:, 2, 0] =  np.zeros(mat_s)
    adjmat[:, 2, 1] =  Ta_sq
    adjmat[:, 2, 2] =  Ta
    detmat =  Ta * Tb_sq - Tb * Ta_sq
    return np.nan_to_num(adjmat / (detmat[:, None, None]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from toolkit.lap.line_normals import linenormals2d
    a = np.arange(150)*0.1
    b = np.square(a)
    c = np.arange(150)*0.01
    d = c**2
    line1 = np.array([a, b])
    line2 = np.array([c, d])
    xyout = linecurvature2D(line1.T)
    # print(xyout)
    xyout1 = linecurvature2D(line2.T)
    # print(xyout1)
    plt.plot(a, b)
    xyout_ln = linenormals2d(line1.T)
    plt.quiver(a, b, xyout_ln[:, 0] * xyout, xyout_ln[:, 1] * xyout)
    plt.plot(c, d)
    plt.show()