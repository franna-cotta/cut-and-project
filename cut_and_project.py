import numpy as np
import numba as nb
from itertools import product, chain
import multiprocessing as mp
import joblib as jl #import Memory, Parallel, delayed
import time 
from math import prod
from random import randrange
import heapq

NUM_CORES = mp.cpu_count()

# Numba decorator for a fast linear combination function
@nb.njit(fastmath=True, cache = True)
def fastLinearCombination(C, B):
    s = np.zeros_like(B[0])
    for i in range(len(C)):
        s += C[i]*B[i]
    return s

class RanProduct:
    def __init__(self, iterables):
        self.its = list(map(list, iterables))
        self.n = prod(map(len, self.its))

    def index(self, i):
        if i not in range(self.n):
            raise ValueError(f"index {i} not in range({self.n})")
        result = []
        for it in reversed(self.its):
            i, r = divmod(i, len(it))
            result.append(it[r])
        return tuple(reversed(result))

    def pickran(self):
        return self.index(randrange(self.n))


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

"""
=== CPS ===
Description: 
Given a lattice, a window, and a method for checking whether points lie in the window
the program computes the resulting cut-and-project set.


Terminology:
Lattice Space - The space in which the higher-dimensional lattice lies

Outer Space - The subset of the lattice space we wish to ultimately project into and obtain output for, typically a 2 or 3-dimensional space.

Inner Space - The subset of the lattice space which the window lies in.

Window - A shape in the outer space for which the intersection with the outer space is taken. 


Inputs:
B : np.array - The basis vectors for the lattice in the form of a basis matrix, where each column is a vector.

outer_dim : int - The dimensions of the inner space (the space to ultimately be projected into)

W - The projection window, in an arbitrary format.

W_offset : np.array - If the window is offset from the centre, we supply its offset as well to shift the generation region of lattice accordingly. 
                      If not specified, the offset is assumed to be zero.

W_check(x) - A function which returns a True value if the point x, when projected into the inner space, lies in a prespecified window W, and a False value if not.

bdry_check(x) - A function which returns True if the point x lies in C, when projected into the outer spave, lies in prespecified boundary region. Used for filtering out the brute force calculations.
                 
N : int = 5 - Search Depth The number of samples to generate from the lattice. The actual number, per dimension, is 2S, generated between +- 5
"""    

def CPS(B : np.array,
        outer_dim : int,
        W_check = None,
        bdry_check = None,
        N : int = 5,
        N_min : int = 0,
        LO : np.array = np.array([], dtype = 'int32'), #lattice offset
        near : np.array = np.array([]),
        debug : bool = True,
        cache_dir : str = "G:\\Joblib\\",
        multithread = False,
        method = ''):
    
    assert 0 <= N_min < N, "N_min must be non-negative and less than N"
    
    # Setting up for joblib later
    num_cores = mp.cpu_count()
    cachedDir = cache_dir
    try:
        mem = jl.Memory(cachedDir)
    except:
        print("Cache directory not found. Use the cache_dir parameter to specify it.")
        return
    
    # Use the j
    FLC = fastLinearCombination
    
    # Lattice dimensions, total dimension of the combined spaces
    dim = 0 

    # Sanity Check: Verify the basis matrix is square and of exactly two dimensions
    if (B.shape[0] != B.shape[1]) or (len(B.shape) != 2):
        print("The basis matrix B must be a square matrix.")
        return 0
    else:
        dim = B.shape[0]

    lattice_size = (2*N + 1)**dim - (2*N_min + 1)**dim
    
    #lattice_size = ((2*(N - N_min) + 1) * (2*N + 1)**(dim-1))*dim
    
    # Sanity Check: Verify the lattice offset is an array of integers
    if LO.size == 0:
        LO = np.zeros(dim, dtype = 'int32')
    else:
        if LO.shape[0] != dim:
            print("Lattice offset vector must be the same dimension as the basis matrix. Assuming a zero vector for now.")
            
            LO = np.zeros(dim, dtype = 'int32')
        if LO.dtype != 'int32':
            LO = LO.astype('int32')
    
    # Sanity Check: Make sure the boundary checking condition exists
    if bdry_check == None:
        bdry_check = lambda x : True
    
    # Sanity Check: Verify the window and a method for checking containment within it have been supplied
    # Could be replaced with just W_check in the future
    if W_check == None:
        print("You must specify a window and check method unless you are using one of the following preprocessing methods: grow_into_window, grow_upto_window, grow_exact_window, adaptive_spherical_window")
        return
    
    # Coordinates in the basis take the form (o_1 , o_2 , o_3 : i_1 , i_2 , i_3 , i_4) where i are the inner dimensiona and o the outer   
    inner_dim = dim - outer_dim  
    
    # Sanity check, to make sure outer_dim < dim
    if outer_dim >= dim:
        print("Degree of inner dimension exceeds or is equal to total dimensions. This means there is no dimensions for the projection space.")
        return 0
    
    # Points in the lattice which lie within the projection
    # Keys correspond to integer coordinates in the lattice basis
    # Values correspond to real coordinates in the standard basis
    found_pts = dict()
    found_pts_outer = dict()
    
    # Shorthand for the function that takes the index vector to the corresponding real space vector
    FLC_lam = lambda x : (x, FLC(x, B.T))
    
    if debug:
        print(f"Starting lattice generation, number of points is {lattice_size}...")
        t0 = time.time()
    
    # Lattice is defined in terms of the integer coordinates
    # TO DO: Add a shift vector parameter so we can offset the origin of the lattice, and choose this shift vector automatically 
    # so that the centre of the lattice lies closed to the centre (or specified offset) of the window
    
    if len(near) == outer_dim:
        curr_origin_index = LO
        curr_origin = FLC(LO, B.T)[:outer_dim]
        curr_dist = np.linalg.norm(curr_origin - near)
        got_closer = True
        
        while got_closer:
            got_closer = False
            local_CPS = CPS(B = B, outer_dim = outer_dim, W_check = W_check, bdry_check = lambda x : True, N = 1, N_min = 0, LO = curr_origin_index, near = np.array([]), debug = False, cache_dir = cache_dir, multithread = False, method = '')
            local_outer_pts = local_CPS[0]
            local_outer_indices = local_CPS[2]
            
            for i in range(len(local_outer_indices)):
                pt = local_outer_pts[i][:outer_dim]
                new_dist = np.linalg.norm(pt - near)
                
                if new_dist < curr_dist:
                    curr_origin_index = np.array(local_outer_indices[i])
                    curr_origin = FLC(curr_origin_index, B.T)[:outer_dim]
                    curr_dist = new_dist
                    got_closer = True
                    break
                    
        LO = curr_origin_index
                    
        print(f"Closest lattice index is {curr_origin_index}")
        
        
    if N_min == 0:
        lattice = [range(- N + LO[j], N + 1 + LO[j]) for j in range(dim)] #(range(-N, N+1),)*dim   
        lattice_iter = product(*lattice)
    else:
        """lattice_iter = list()
        
        for i in range(dim):   
            lattice_start = [range(- N + LO[j], N + 1 + LO[j]) for j in range(i)]
            lattice_middle = [chain(range(- N + LO[i], - N_min + 1 + LO[i]), range(N_min + LO[i], N + 1 + LO[i])),]
            lattice_end = [range(- N + LO[j], N + 1 + LO[j]) for j in range(i + 1, dim)]
            
            lattice = lattice_start + lattice_middle + lattice_end
            lattice_iter.append(product(*lattice))
            
        lattice_iter = chain(*lattice_iter)"""
        
        lattice = [range(- N + LO[j], N + 1 + LO[j]) for j in range(dim)]
        ignored_lattice = [range(- N_min + LO[j], N_min + 1 + LO[j]) for j in range(dim)]
        
        if lattice_size <= 10e6 or method == 'set':
            if debug:
                print("Using set lookup method")
                print(f"Starting set conversion, number of points is {(2*N_min + 1)**dim}...")
                t00 = time.time()
            ignored_indices = set(product(*ignored_lattice))
            
            if debug:
                print(f"Finished set conversion, took {time.time() - t00:.2f} seconds.")
        
            lattice_filter = lambda x : x not in ignored_indices
        else:
            if debug:
                print("Using size filtering method")
            lattice_filter = lambda x : any(abs(y) >= N_min for y in x)
        
        lattice_iter = filter(lattice_filter, product(*lattice))
        #lattice_iter = filter(lambda x : np.any(np.abs(np.array(x)) >= N_min), product(*lattice))
        
    debug_pts = 0
                
    if multithread:
        if isinstance(multithread, int):
            n = multithread
        else:
            n = NUM_CORES - 1
        
        parallel = jl.Parallel(n_jobs = NUM_CORES - 1, return_as = 'generator_unordered', verbose = 5, batch_size = 1024, prefer = 'threads')
        #lattice_gen = parallel(jl.delayed(FLC_lam)(idx) for idx in filter(lambda x : np.any(np.abs(np.array(x)) >= N_min), lattice_iter))
        lattice_gen = parallel(jl.delayed(FLC_lam)(idx) for idx in lattice_iter)
    else:
        #lattice_gen = map(FLC_lam, filter(lambda x : np.any(np.abs(np.array(x)) >= N_min), lattice_iter))
        lattice_gen = map(FLC_lam, lattice_iter)
    #lattice_gen = product(*lattice)
    
    debug_ctr = 0
    
    """ Iterate through all possible lattice points in the sample  and check whether they're in the window.
    This is a crude, brute-force method. Uses O((2N)**dim) loops in a worst-case scenario."""    
    for idx, pt in lattice_gen: # product(*lattice):
        debug_ctr += 1
        # Check if the inner dimensions have already been found and skip the whole iteration if so
        ##if tuple(idx[:inner_dim]) in lower_idx_found:
        #    continue
        
        # Calculate the lattice point by taking the linear combination of the integer coefficients with the basis vector
        #pt = FLC(idx, B.T)

        # Slice to obtain the canonical projections into the inner and outer dimensions
        # TO DO: Generalize this to accept non-canonical projection mappings
        pt_outer = pt[:inner_dim]
        pt_inner = pt[outer_dim:]

        # Check if its inner coordinates lies in the window and its outer coordinates lie in the boundary
        if bdry_check(pt_outer) and W_check(pt_inner):
            # The (real) points located in the window are indexed by their integer coefficients in the lattice
            #lower_idx_found.append(tuple(idx[:inner_dim]))
            found_pts[tuple(idx)] = pt
    
    # Create a version of found_pts that consists of the projected-down veryion of the points
    found_pts_outer = {tuple(p[:outer_dim]) : k for (k, p) in found_pts.items()}

    if debug:
        print(f"True length of lattice is {debug_ctr}")
    
    if debug:
        print(f"Finished generating lattice, took {time.time() - t0:.2f} seconds.")
    
    # Point (outer real coords), points (real coords), points (lattic coords), #edges (real coords), edges (lattice coords)
    return (list(found_pts_outer.keys()),
            list(found_pts.values()),
            list(found_pts.keys()))

class CPS_iterator:
    def __iter__(self):
        return self

    def __next__(self):
            try:
                while True:
                    idx, pt = next(self.iter)
                
                    # Slice to obtain the canonical projections into the inner and outer dimensions
                    # TO DO: Generalize this to accept non-canonical projection mappings
                    pt_outer = pt[:self.inner_dim]
                    pt_inner = pt[self.outer_dim:]
            
                    # Check if its inner coordinates lies in the window and its outer coordinates lie in the boundary
                    if self.bdry_check(pt_outer) and self.W_check(pt_inner):
                        # The (real) points located in the window are indexed by their integer coefficients in the lattice
                        self.current = (tuple(idx), pt[:self.outer_dim])
                        break
                        #return pt[:self.outer_dim]
    
            except StopIteration:
                self.current = None
    
            finally:
                return self.current
    
    def __init__(self, B : np.array,
        outer_dim : int,
        W_check = None,
        bdry_check = None,
        cache_dir : str = "G:\\Joblib\\"):

        self.current = None
    
        # Setting up for joblib later
        cachedDir = cache_dir
        try:
            mem = jl.Memory(cachedDir)
        except:
            print("Cache directory not found. Use the cache_dir parameter to specify it.")
            return
        
        self.dim = B.shape[0]
        
        # Sanity Check: Make sure the boundary checking condition exists
        if bdry_check == None:
            self.bdry_check = lambda x : True
        else:
            self.bdry_check = bdry_check
        
        # Sanity Check: Verify the window and a method for checking containment within it have been supplied
        # Could be replaced with just W_check in the future
        if W_check == None:
            raise Exception("You must specify a window and check method unless you are using one of the following preprocessing methods: grow_into_window, grow_upto_window, grow_exact_window, adaptive_spherical_window")
        else:
            self.W_check = W_check
        
        # Coordinates in the basis take the form (o_1 , o_2 , o_3 : i_1 , i_2 , i_3 , i_4) where i are the inner dimensiona and o the outer   
        self.inner_dim = self.dim - outer_dim
        self.outer_dim = outer_dim
        
        # Sanity check, to make sure outer_dim < dim
        if self.outer_dim >= self.dim:
            raise Exception("Degree of inner dimension exceeds or is equal to total dimensions. This means there is no dimensions for the projection space.")
    
        # Lattice is defined in terms of the integer coordinates
        # we sort so that the iteration starts from the origin and spreads outwards
        lattice = (sorted(range(-100, 101), key = lambda x : np.sum(np.abs(x))), )*self.dim
        FLC_lam = lambda x : (x, fastLinearCombination(x, B.T))  
        
        self.iter = map(FLC_lam, nearest_first_product(*lattice))
        