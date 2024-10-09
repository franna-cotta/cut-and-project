import numpy as np
import numba as nb
from itertools import product, islice, combinations
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from shapely.affinity import translate, rotate
from scipy.spatial import Delaunay, ConvexHull
from joblib import Memory
import random

def trueLambda(W, x):
    return True

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

@nb.njit(fastmath=True)
def fastNorm(l):    
    s = 0.
    for x in l:
        s += x**2
    return np.sqrt(s)

@nb.njit(fastmath=True)
def fastSquaredNorm(l):
    s = 0.
    for x in l:
        s += x**2
    return s

@nb.njit(fastmath=True)
def fastLinearCombination(C, B):
    s = np.zeros_like(B[0])
    for i in range(len(C)):
        s += C[i]*B[i]
    return s

@nb.njit(fastmath = True)
def nearestVectorEnumerate(p, L):
    leastDist = np.dot(p - L[0])
    nearest = 0
    
    for i in range(len(L)):
        dist = np.dot(p - L[i])
        if dist < leastDist:
            leastDist = dist
            nearest = i
            
    return nearest

# Takes a point x and a list of length-2 tuples R
def is_in_interval(R, x):   
    assert len(R) == len(x), "Error, the dimensions of the points do not match the interval."
    
    for i in range(len(x)):
        if x[i] < R[i][0] or x[i] > R[i][1]:
            return False
            
    return True

# Takes a point x and a list of length-2 tuples R
point_in_interval = lambda R, x : R[0] <= x <= R[1]  

def point_in_ncube(R, pt):
    for x in pt:
        if not point_in_interval(R, x):
            return False
            
    return True

# For each coordinate of x, checks it lies in the hypercube defined by R
def pts_in_ndim_interval(R, pts):
    return [x for x in pts if is_in_interval(R, x)]

def delaunay_hull_point_check(hull, p):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    
    #if np.linalg.norm(p) > max(np.linalg.norm(q) for q in hull.points):
   #    return False

    #else:
    return hull.find_simplex(p) >= 0

def convex_hull_point_check(hull, p, tolerance=1e-12):
    if not isinstance(hull,ConvexHull):
        hull = ConvexHull(hull)
        
    return all( (np.dot(eq[:-1], p) + eq[-1] <= tolerance) for eq in hull.equations )

def delaunay_hull_point_check_generator(dim1 : int = 0, dim2 : int = 1):
    def inner_fn(hull, p):
        q = (p[dim1], p[dim2])
        
        if not isinstance(hull,Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(q) >= 0 
    
    return inner_fn

def delaunay_regular_polygon(n : int, length : float = 1.0, length_type : str = 'c', offset : np.array = np.array([0,0]), rotation : float = 0.0):   
    if length_type == 'c': # Circumradius
        v0 = np.array([0.0, length])
    elif length_type == 'e': # Edge-length
        v0 = np.array([0.0, length/(2*np.sin(np.pi/n))])
    elif length_type == 'a': # Apothem
        v0 = np.array([0.0, length/(np.cos(np.pi/n))])
    else: # Default to circumradius
        v0 = np.array([0.0, length])
    
    theta = 2.0*np.pi/n
    
    Rp = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]        
    ])
    
    Rf = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]        
    ])
    
    points = [np.dot(Rf, np.linalg.matrix_power(Rp,i).dot(v0)) + offset for i in range(n)]
    
    return Delaunay(points)

def shapely_regular_polygon(n : int, length : float = 1.0, length_type : str = 'c', offset : tuple = (0,0,0), rotation : float = 0.0, stellation : float = 0.0):

    if length_type == 'c': # Circumradius
        v0 = np.array([0.0, length])
    elif length_type == 'e': # Edge-length
        v0 = np.array([0.0, length/(2*np.sin(np.pi/n))])
    elif length_type == 'a': # Apothem
        v0 = np.array([0.0, length/(np.cos(np.pi/n))])
    else: # Default to circumradius
        v0 = np.array([0.0, length])
    
    theta = 2.0*np.pi/n
    
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]        
    ])
    
    W = Polygon([(1.0 + (stellation * (i%2)))*np.linalg.matrix_power(R,i).dot(v0) for i in range(n)])
    W = rotate(W, rotation, use_radians = True)
    W = translate(W, xoff = offset[0], yoff = offset[1], zoff = offset[2])
    
    return W

shapely_point_in_poly_check = lambda S, p : S.contains(Point(p))

# In cases where the inner dimensions are > 2 we need to specify which 2-axis plane the polygon is projected into.
def shapely_point_in_poly_check_generator(dim1 : int = 0, dim2 : int = 1):
    return lambda S, p : S.contains(Point((p[dim1], p[dim2])))

# In cases where the inner dimensions are > 2 we need to specify which 2-axis plane the polygon is projected into.
def shapely_point_in_multipoly_check_generator(dim1 : int = 0, dim2 : int = 1):
    return lambda S, p : S.contains(Point((p[dim1], p[dim2])))

point_in_nsphere = lambda r, p : fastNorm(p) <= r

@nb.njit(fastmath=True)
def point_squared_in_nsphere (r, p, O = None):
    #p = np.array(p)
    
    #if not O:
        #O = np.array((0,)*len(p))
        #O = [(0,)*len(p)]
    
    i = 0
    s = 0.
    while i < len(O):
        s += (p[i] - O[i])**2
        
        if s > r**2:
            return False
        
        i += 1
        
    return True
    
    #return fastSquaredNorm(p - O) <= r**2

@nb.njit(fastmath=True)
def point_squared_in_nsphere_prog(r : float, V : tuple):
    # List of coefficients and list of basis vectors
    C, B = V[0], V[1]
    
    v = fastLinearCombination(C, nb.typed.List(B))
    
    if len(C) != len(B): # both vectors need to have same dimension
        return False

    current_sum = 0.0

    #if np.any(v < -r) or np.any(v > r):
    #    return False
    
    for i in range(len(v)):          
        current_sum += v[i]**2
        
        if current_sum > r**2: #any additional points will further increase distance from Origin
            return False 

    return True # The point is within the sphere of radius 'r' 

def post_process_merge_clusters(points, d, loop : int = 0):
    def dist2(p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    
    ret = []
    d2 = d * d
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n):
                if dist2(points[i], points[j]) < d2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count+=1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            ret.append(np.array([point[0], point[1]]))
            
    if loop > 0:
        return post_process_merge_clusters(ret, d, loop = loop - 1)
    else:
        return ret

# Takes two point sets in the form of lists of vectors and returns the minkowski difference L1 + L2
def minkowski_set_sum(L1, L2, form : str = ''):
    if form == 'tuple':
        return [(a + b, a, b) for a, b in product(L1, L2)]
    elif form == 'index':
        return [(L1[i] + L2[j], i, j) for i, j in product(range(len(L1)), range(len(L2)))]
    else:
        return [a + b for a, b in product(L1, L2)]

# Takes two point sets in the form of lists of vectors and returns the minkowski difference L1 - L2
def minkowski_set_difference(L1, L2, form : str = '', norm = 2):
    L1, L2 = np.array(L1), np.array(L2)
    
    if form == 'tuple':
        return [(a - b, a, b) for a, b in product(L1, L2)]
    elif form == 'index':
        return [(L1[i] - L2[j], i, j) for i, j in product(range(len(L1)), range(len(L2)))]
    elif form == 'normindex':
        return [(np.linalg.norm(L1[i] - L2[j], ord = norm), i, j) for i, j in product(range(len(L1)), range(len(L2)))]
    else:
        return [a + b for a, b in product(L1, L2)]

# Return true if the two line segments share any endpoints
def share_endpoints(e1, e2):
    return (np.array_equal(e1[0], e2[0])
            or np.array_equal(e1[0], e2[1])
            or np.array_equal(e1[1], e2[0])
            or np.array_equal(e1[1], e2[1]))
    
# Using the method outline in Aperiodic Order Vol 1, Theorem 6.1., page 185
# Not sure if it generalizes
def recover_edges(LAMBDA, prec : int = 4, order = 0, norm = 2, remove_overlapping : str = ''):    
    # Calculate the Minkowski difference
    L_norm = minkowski_set_difference(LAMBDA, LAMBDA, 'normindex', norm)
    
    # Sort the points in order of ascending norm
    L_norm.sort(key = lambda p : p[0])
    
    # Collect all the unique distances (rounded to some precision)
    # Ensure they're unique by forming a set
    # Then convert back to a list and sort ascending
    L_distances = sorted(list(set([np.round(p[0], decimals = prec) for p in L_norm])))
    print(L_distances)
    
    # We will now collect and group all the edges by which distance group they fall into
    L_edges_by_distance = list()
    
    if type(order) == int:
        distances = [L_distances[order]]
    elif order == 'all':
        distances = L_distances
    else:
        distances = [L_distances[o] for o in order]
        
    print(distances)
    
    found_edges = list()
    remove_at_end = set()
    
    # Pre-cache all the edges
    for t in L_norm:            
        # If the norm matches the a distance, then the edge is found
        if np.round(t[0], decimals = prec) in distances:
            e1 = (LAMBDA[t[1]], LAMBDA[t[2]])

            # Check for duplicates already in the list
            # If none are found, proceed
            if not any(np.array_equal(e1, e2) for e2 in found_edges):
                # For non-intersection handling or multi-intersection handling, we need the whole list unmodified
                if remove_overlapping == '' or remove_overlapping == 'multi':
                    found_edges.append(e1)
                    
                # For single-intersection handling, we check to see if the new edge intersects any
                # existing edges. If it does, we don't add it
                elif remove_overlapping == 'single':
                    overlap = False
                    
                    for e2 in found_edges:
                        L1, L2 = LineString(e1), LineString(e2)
                        
                        if L1.intersects(L2) and not share_endpoints(e1, e2):
                            overlap = True
                            
                    if not overlap:
                        found_edges.append(e1)
                        
                            
    N_edges = len(found_edges)
    
    # Overlap checking
    if remove_overlapping == 'multi':
        # Iterate through all pairs of edges
        for i, j in product(range(N_edges), repeat = 2):
            e1, e2 = found_edges[i], found_edges[j]
            L1, L2 = LineString(e1), LineString(e2)
            
            # Flag
            overlap = False

            # Check if the e1 intersects e2 and that they don't have any endpoints in common
            if L1.intersects(L2) and not share_endpoints(e1, e2):                  
                remove_at_end = remove_at_end.union({i, j})
                  
        # Filter out the overlapping points
        return [found_edges[k] for k in range(N_edges) if (k not in remove_at_end)]
                
    return found_edges

# Computes the area of a triangle given its three coordinates
def shoelace_area(pts):
    px = pts[:,0]
    py = pts[:,1]
    i = np.arange(len(px))
    
    return np.abs(np.sum(px[i-1]*py[i]-px[i]*py[i-1])*0.5)

import heapq

# https://stackoverflow.com/questions/42288203/generate-itertools-product-in-different-order
def nearest_first_product(*sequences):
    start = (0,)*len(sequences)
    queue = [(0, start)]
    seen = set([start])
    while queue:
        priority, indexes = heapq.heappop(queue)
        yield tuple(seq[index] for seq, index in zip(sequences, indexes))
        for i in range(len(sequences)):
            if indexes[i] < len(sequences[i]) - 1:
                lst = list(indexes)
                lst[i] += 1
                new_indexes = tuple(lst)
                if new_indexes not in seen:
                    new_priority = sum(index * index for index in new_indexes)
                    heapq.heappush(queue, (new_priority, new_indexes))
                    seen.add(new_indexes)

# Taken from https://code-examples.net/en/q/1434a3b
# 'Shuffles' an iterator by taking chunks of buffer_size and shuffling those
def iterShuffler(generator, buffer_size):
    while True:
        buffer = list(islice(generator, buffer_size))
        if len(buffer) == 0:
            break
        np.random.shuffle(buffer)
        for item in buffer:
            yield item

## Shell iterator
# Starting with an initial point, iterates through the cartesian product
# as if constructing it as a sequence of shells around the initial point
# and adding one layer at a time.
def shellIterator(origin, limit : int = 1000, shuffleBuffer : int = 0):
    # The origin is assumed to be a list, numpy array or other iterable which acts as the starting point for the growth
    origin = np.array(origin)
    dim = origin.shape[0]
    currLoop = 1
    prevLoops = {0}

    innerIterator = product((-1, 0, 1), repeat = dim)
    
    v = origin.copy()
    
    while True:
        if currLoop > limit:
            return
            
        yield v
        
        try:
            m = next(innerIterator)
            
        except StopIteration:
            prevLoops = prevLoops.union({-currLoop, currLoop})
            currLoop += 1
            
            innerIterator = product(range(-currLoop, currLoop+1), repeat = dim)
            
            if shuffleBuffer > 0:
                innerIterator = iterShuffler(innerIterator, buffer_size = shuffleBuffer)
            
            m = next(innerIterator)
        
        while all(x in prevLoops for x in m):
            try:
                m = next(innerIterator)
            except StopIteration:
                prevLoops = prevLoops.union({-currLoop, currLoop})
                currLoop += 1
                
                innerIterator = product(range(-currLoop, currLoop+1), repeat = dim)
                
                if shuffleBuffer > 0:
                    innerIterator = iterShuffler(innerIterator, buffer_size = shuffleBuffer)
                    
                m = next(innerIterator)
        
        else:
            v = origin + m
            
# Ring Iterator
# modification of Shell Iterator that does not grow outwards
def ringIterator(origin, ring : int = 1, shuffleBuffer : int = 0, basis = None, maxIter = False):
    assert ring >= 1, "Must specify a positive, integer ring number"
    
    # The origin is assumed to be a list, numpy array or other iterable which acts as the starting point for the growth
    O = np.array(origin)
    dim = int(O.shape[0])

    # If a basis is provided, we generate the ranges in proportion to the scale of the basis vectors
    if np.any(np.array(basis)):
        basis = np.array(basis)
        assert basis.shape[0] == dim, "Supplied basis dimensions must match those of the origin"
    else:
        basis = np.diag((1,)*dim)
    
    maxNorm = max(fastNorm(b) for b in basis)
    
    # Collect the norms of the basis vectors, normalized so the largest is 1
    R = [ring * int(np.ceil(maxNorm / fastNorm(b))) for b in basis]
    
    if maxIter:
        Rmax = int(max(R))
        R = [Rmax for b in basis]
    
    ranges = [sorted(range(-R[i] + O[i] , 1 + R[i] + O[i]), key = lambda x : np.abs(x - O[i])) for i in range(dim)]
    prevLoops = [set(range(1 - R[i] + O[i] , R[i] + O[i])) for i in range(dim)]
    
    innerIterator = nearest_first_product(*ranges) #product(*ranges)
    
    if shuffleBuffer > 0:
        innerIterator = iterShuffler(innerIterator, buffer_size = shuffleBuffer)

    while True:
        try:
            m = next(innerIterator)
                
        except StopIteration:
            return
        
        # Check this condition tomorrow
        while all(all((x in S) for x in m) for S in prevLoops): #all(x in prevLoops for x in m): 
            try:
                m = next(innerIterator)

            except StopIteration:
                return

        yield m
        
def tri_similar(T1, T2, prec = 8):
    V = [T1[1] - T1[0], T1[2] - T1[1], T1[0] - T1[2]]
    W = [T2[1] - T2[0], T2[2] - T2[1], T2[0] - T2[2]]
    
    VA = [np.round(np.dot(v1, v2)/(np.dot(v1, v1) * np.dot(v2, v2)), decimals = prec) for v1, v2 in combinations(V, r=2)]
    WA = [np.round(np.dot(w1, w2)/(np.dot(w1, w1) * np.dot(w2, w2)), decimals = prec) for w1, w2 in combinations(W, r=2)]
    
    contain_count = 0
    
    for a in WA:
        if a in VA:
            contain_count += 1
            
    if contain_count >= 2:
        return True
    else:
        return False
    