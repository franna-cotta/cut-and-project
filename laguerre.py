import numpy as np
import csv
import subprocess
import tempfile
import copy

def is_float_try(str):
    try:
        float(str)
        return True
    except ValueError:
        return False
    
class laguerre_tess:
    # Path to the powerdiagram executable
    powerdiagram_path = "./Executables/powerdiagram.exe"
    
    def __init__(self, pts = list(), rad = 1.0):
        self.centres = pts
        
        if isinstance(rad, (float, int)):
            self.radii = [rad for i in range(len(pts))]
            
        elif isinstance(rad, (list, tuple)):
            if len(rad) != len(pts):
                raise Exception("If supplied as a list, the number of radii must match the number of centres.")
            else:
                self.radii = [r for r in rad]
        else:
            raise TypeError("Radii must either be specified as a scalar value or as a list of values of equal length as the list of centres.")
            
        self.centres_file = None
        self.radii_file = None

    def _rescale_data(self):
        return 0
       
    def load_from_csv(self, centres_file, radii_file):
        with open(centres_file, 'r', newline='') as centres_csv, open(radii_file, 'r', newline='') as radii_csv:
            centres_reader = csv.reader(centres_csv, delimiter = ',')
            radii_reader = csv.reader(radii_csv, delimiter = ',')
            
            self.centres = [[float(x) for x in row] for row in centres_reader]
            self.radii = [float(x[0]) for x in radii_reader]
            
        self.centres_file = centres_file
        self.radii_file = radii_file
        
        print(f"CSV data read successfully! Number of centres and radii found is {len(self.centres)}.")
        
    def generate(self, naive = False, dual = False, debug = False):
        if self.centres_file == None:
            # Create temporary directories for the CSV files
            centres_csv_tmp = tempfile.NamedTemporaryFile(suffix = '.csv', delete = False)
            radii_csv_tmp = tempfile.NamedTemporaryFile(suffix = '.csv', delete = False)

            with open(centres_csv_tmp.name, 'w', newline='') as centres_csv, open(radii_csv_tmp.name, 'w', newline='') as radii_csv:
                # Write the centres and radii to disk
                centres_writer = csv.writer(centres_csv)
                radii_writer = csv.writer(radii_csv)

                # Convert to high-precision strings
                c_str = [[f"%.3f" % c for c in coords] for coords in self.centres]
                r_str = [f"%.3f" % r for r in self.radii]

                centres_writer.writerows([c for c in c_str])
                radii_writer.writerows([(r,) for r in r_str])
                
            centres_csv, radii_csv = centres_csv_tmp.name, radii_csv_tmp.name
        else:
            centres_csv, radii_csv = self.centres_file, self.radii_file

        # Empty the existing data
        la_pts, la_edges_idx, la_edges, la_rays, la_rays_idx = list(), list(), list(), list(), list()
        
        pd_args = [laguerre_tess.powerdiagram_path,]
        if naive:
            pd_args.append('-naive')
        if dual:
            pd_args.append('-dual')
            
        pd_args += ['-draw', centres_csv, radii_csv]
        
        print(self.centres_file)
        
        # Run the Powerdiagram executable with the -draw parameter and capture the output
        pd = subprocess.run(pd_args, capture_output = True)

        # Check the executable ran correctly
        pd.check_returncode()

        # Decode the bytes
        pd = pd.stdout.decode("utf-8")
        
        if debug:
            print(pd)

        # Split into spheres, points, and edges by interpreting the output from powerdiagram
        try:
            pd_spheres, pd_pts, pd_edges = pd.split("\r\n\r\n")
        except:
            raise Exception("Something went wrong!")

        # Split each into a list by line break
        pd_spheres, pd_pts, pd_edges = pd_spheres.split('\r\n'), pd_pts.split('\r\n'), pd_edges.split('\r\n')
        
        # Pop off the final, empty element
        pd_edges.pop(-1)
        
        print(f"Number of spheres found {len(pd_spheres)}")
        print(f"Number of points found {len(pd_pts)}")
        print(f"Number of edges found {len(pd_edges)}")
              
              
        # Points are expressed as strings of the form
        # p1 0.625 0.5
        for pt in pd_pts:
            # The points arrive in the form of strings
            # 'p1 0.34343 -0.23114'
            # We segment out the floats and store them as a a list of pairs
            la_pts.append([float(s) for s in pt.split(' ') if is_float_try(s)])

            
        # Edges are expressed as strings of either the form
        # ei p1 p3 s1 s4
        # ee p1 s1 s3 d-0.6 0.8
        for ed in pd_edges:
            # Edges come in two varieties, edges between points (internal edges, ei) and rays directed from a single point (external edges, ee)
            
            # Internal edges will always have two 'p' entries
            if ed[:2] == 'ei':            
                # For internal edges we just split out the strings that start with a p to get the indices of the points connected
                # then convert them to an int
                # The internal indices are indexed starting at 1, so we have to subtract 1
                ei = [int(s[1:]) - 1 for s in ed.split(' ') if s != '' and s[0] == 'p']

                la_edges_idx.append(ei)

            elif ed[:2] == 'ee':
                # We will deal with rays at a later date
                ee_origin_idx = [int(s[1:]) - 1 for s in ed.split(' ') if s != '' and s[0] == 'p']
                
                # Convert to real coordinates
                ee_origin = la_pts[ee_origin_idx[0] - 1]
                
                ee_dir_str = ed[ed.find('d') + 1 : ].split(' ')
                
                ee_dir = [float(x) for x in ee_dir_str if x != '']
                
                """if ee_dir_str[0] == '-0':
                    ee_dir[1] *= -1
                if ee_dir_str[1] == '-0':
                    ee_dir[0] *= -1"""
                
                ee_terminus = [a + b for a, b in zip(ee_origin, ee_dir)]
                
                la_rays.append((ee_origin, ee_terminus))
                la_rays_idx.append((ee_origin_idx[0], ee_dir))

            else:
                pass

                
        print(f"Number of (internal) edges found {len(la_edges_idx)}")
                
        # Create float versions of the edges
        la_edges = [(la_pts[i-1], la_pts[j-1]) for i, j in la_edges_idx]

        print("Generation complete!")
        
        return {"Centres" : self.centres,
                "Radii" : self.radii,
                "LVerts" : la_pts,
                "LEdgesIdx" : la_edges_idx,
                "LEdges" : la_edges,
                "LRays" : la_rays,
                "LRaysIdx" : la_rays_idx}
    

# Take an input consisting of a matplotlib plot (P) and a Laguerre tiling (LA) and returns various plots
# Currently not working after adding in a -1 offset to the eddge indices
def plot_laguerre(P, LA, plot_verts = False, plot_points = False, plot_edges = True, plot_rays = True):
    fig, ax = P.subplots(figsize = [15,15])
    
    pts, la_pts, la_edges, la_rays = LA["Centres"], LA["LVerts"], LA["LEdges"], LA["LRays"]
    
    xmin = min([x[0] for x in pts])
    xmax = max([x[0] for x in pts])
    ymin = min([x[1] for x in pts])
    ymax = max([x[1] for x in pts])
    
    if plot_points:
        for p in pts:
            P.scatter(p[0], p[1], color = 'C1')
    
    if plot_verts:
        for p in la_pts:
            P.scatter(p[0], p[1], color = 'C0')
    
    if plot_edges:
        for e in la_edges:
            P.plot((e[0][0], e[1][0]), (e[0][1], e[1][1]), color = 'C0')
  
    if plot_rays:
        for r in la_rays:
            P.plot((r[0][0], r[1][0]), (r[0][1], r[1][1]), color = 'C3')

    P.xlim([xmin, xmax])
    P.ylim([ymin, ymax])
    P.show()