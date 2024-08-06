from numpy import array

G = 9.81
PI = 3.141592653589793
FTLB_TO_NM = 1.3558 # foot pounds to newton meters - torque
INLB_TO_NM = 0.1129848 # inch pounds to newton meters - torque
IN_TO_M = 1/39.37 # inches to meters
IN_TO_MM = 25.4 # inches to millimeters
LB_TO_KG = 1/2.2 # pounds to kag
KMH_TO_MS = 0.44704 # km per hr to meters per sec
MS_TO_MPH = 2.23694 # meters per sec to miles per hr
RPM_TO_RAD = 2*3.1415/60 # rotations per min to rad per sec
PSI_TO_PA = 6894.76 # pounds per square inch to pascals
NVEC = array([1/G, 1/G, 1/10.0]) # normal vector for solving point error distance
NVEC2 = array([1/G, 1/G, 1/10.0, 1.0, 1.0]) # normal vector for solving point error distance
MIRROR = array([1.0, -1.0, -1.0, -1.0, -1.0])
STRAIGHT = array([1.0, 1.0, 1.0, 1.0, 1.0])