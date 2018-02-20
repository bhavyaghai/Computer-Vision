# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
from matplotlib import pyplot as plt
import maxflow
from scipy.spatial import Delaunay


drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve
color = 'FR' # FR means foreground & BB means background
#pts_f = []
#pts_b = []

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=650, compactness=20)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        # The indices of neighboring vertices of vertex k are indptr[indices[k]:indices[k+1]].
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

# mouse callback function
def mouse_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode #, pts_f, pts_b

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
        #pts_f.append((current_former_x, current_former_y))
        #print(current_former_x, current_former_y)

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                if color == 'FR':
                    cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
                elif color == 'BB':
                    cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,0,0),5)
                current_former_x = former_x
                current_former_y = former_y
                #if color == 'FR':
                #    pts_f.append((current_former_x, current_former_y))
                #elif color == 'BB':
                #    pts_b.append((current_former_x, current_former_y))
                #print(current_former_x, current_former_y)
            
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            if color == 'FR':
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),5)
            elif color == 'BB':
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(255,0,0),5)
            current_former_x = former_x
            current_former_y = former_y
            #if color == 'FR':
            #    pts_f.append((current_former_x, current_former_y))
            #elif color == 'BB':
            #    pts_b.append((current_former_x, current_former_y))
            #print(current_former_x, current_former_y)

    return former_x,former_y    


def generate_marking(color_hists, superpixels, neighbors, norm_hists, img_marking, img):
    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)
    fg_hist = cumulative_histogram_for_superpixels(fg_segments, norm_hists)
    bg_hist = cumulative_histogram_for_superpixels(bg_segments, norm_hists)
    fgbg_hists = [fg_hist, bg_hist]
    fgbg_superpixels = [fg_segments, bg_segments]
    # graph_cut is a boolean array where each distinct value represnts each category 
    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
    mask = np.zeros(shape=(img.shape[0],img.shape[1])) 
    # getting all true values or superpixel ids of either fg/bg
    ids = graph_cut.nonzero()[0]
    for i in ids:
        mask[superpixels==i] = 1
    mask = np.uint8(mask * 255)
    #Validate result RMSE value
    key = cv2.imread("example_output.png", 0)
    #print(mask)
    #print(RMSD(mask, key))
    # ======================================== #
    #output_name = sys.argv[3] + "mask.png"
    #cv2.imwrite(output_name, mask);
    return mask


if __name__ == '__main__':
   
    # validate the input arguments
    print "Press f for foreground\n"
    print "Press b for background\n"
    print "Press c to clear lines\n"
    print "Press [Enter] for Segmentation\n"
    im = cv2.imread(sys.argv[1], 1)
    # ======================================== #
    # write all your codes here
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(im)
    norm_hists = normalize_histograms(color_hists)

    cv2.namedWindow("CV HW4 - Bhavya Ghai")
    cv2.setMouseCallback('CV HW4 - Bhavya Ghai',mouse_draw)
    print("Foreground Selected")

    flag = 0 # output ready
    marking = np.zeros((im.shape[1],im.shape[0],3), np.uint8)
    marking.fill(255)
    while(1):
        cv2.imshow('CV HW4 - Bhavya Ghai',im)
        if flag==1:
            cv2.imshow('Segmentation',output)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
        elif k==ord('f'):
            print("Foreground Selected")
            color = 'FR'
        elif k==ord('b'):
            color = 'BB'
            print("Background Selected")    
        elif k==ord('c'):
            im = cv2.imread(sys.argv[1], 1)
            marking = np.zeros((im.shape[1],im.shape[0],3), np.uint8)
            marking.fill(255)
            cv2.destroyWindow('Segmentation')
            print('All lines cleared')
            flag = 0
            #pts_b = []
            #pts_f = []
        elif k==13:
            #for i in pts_f:
        #    marking[i[1],i[0]] = (0,0,255)
        #for i in pts_b:
        #    marking[i[1],i[0]] = (255,0,0)
            orig = cv2.imread(sys.argv[1], 1)
            ind = np.array(np.where(im!=orig)).T
            #c,c1 = 0,0
            #print(ind.shape)
            for i in ind:
                x,y = i[0],i[1]
                if list(im[x,y])== [0,0,255]:
                    marking[x,y] = [0,0,255]
                    #c = c+1
                elif list(im[x,y])== [255,0,0]:
                    marking[x,y] = [255,0,0]
                    #c = c+1
                else:
                    print orig[x,y],"      ",im[x,y] 
            output = generate_marking(color_hists, superpixels, neighbors, norm_hists, marking, im)
            flag = 1
                    #c1 = c1+1
                #print im[x,y]
        #marking = marking + (im-orig)
        #print ind.shape[0],c,c1
        #cv2.imshow('marking', marking)
        #cv2.waitKey(0)
        #print("Show segemented Output")

    cv2.destroyAllWindows()
    

    