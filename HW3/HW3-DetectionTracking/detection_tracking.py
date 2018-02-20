# Bhavya Ghai
# 111168954
# python detection_tracking.py 2 02-1.avi .\output\

import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('C:\Users\Bhavya\Anaconda2\pkgs\opencv3-3.1.0-py27_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

def help_message():
   print("Usage: [Question_Number] [Input_Video] [Output_Directory]")
   print("[Question Number]")
   print("1 Camshift")
   print("2 Particle Filter")
   print("3 Kalman Filter")
   print("4 Optical Flow")
   print("[Input_Video]")
   print("Path to the input video")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " 1 " + "02-1.avi " + "./")


def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0,0,0,0)
    return faces[0]

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist


def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

def camshift_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    #print(c,r,w,h)

    # Write track point for first frame
    pt = (frameCounter,c+w/2,r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1 )

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        c,r,w,h = track_window
        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        #cv2.circle(frame, (c+w/2,r+h/2), 3, (0,0,255), -1)
        #img2 = cv2.polylines(frame,[pts],True, 255,2)
        #cv2.imshow('img2',img2)
        #cv2.waitKey(60) & 0xff
    
        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2)) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    #cv2.destroyAllWindows()
    output.close()


# a function that, given a particle position, will return the particle's "fitness"
def particleevaluator(back_proj, particle):
    return back_proj[particle[1],particle[0]]


def particle_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)

    # Write track point for first frame
    pt = (frameCounter,c+w/2,r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    # calculate the HSV histogram in the window
    # NOTE: you do not need this in the Kalman, Particle or OF trackers
    roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for you

    n_particles = 200
    init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
    particles = np.ones((n_particles, 2), int) * init_pos

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    f0 = particleevaluator(hist_bp, init_pos) * np.ones(n_particles) # Evaluate appearance model
    weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #roi_hist = hsv_histogram_for_window(frame, (c,r,w,h))
        hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # Particle motion model: uniform step (TODO: find a better motion model)
        stepsize = 10
        np.add(particles, np.random.uniform(-stepsize, stepsize, particles.shape), out=particles, casting="unsafe")

        # Clip out-of-bounds particles
        particles = particles.clip(np.zeros(2), np.array((frame.shape[1],frame.shape[0]))-1).astype(int)

        f = particleevaluator(hist_bp, particles.T) # Evaluate particles
        weights = np.float32(f.clip(1))             # Weight ~ histogram response
        weights /= np.sum(weights)                  # Normalize w
        pos = np.sum(particles.T * weights, axis=1).astype(int) # expected position: weighted average

        if 1. / np.sum(weights**2) < n_particles / 2.: # If particle cloud degenerate:
            particles = particles[resample(weights),:]  # Resample particles according to weights
        # resample() function is provided for you

        pos = np.sum(particles.T * weights, axis=1).astype(int)
        # if you track particles - take the weighted average
        #for i in range(particles.shape[0]):
        #    cv2.circle(frame, (particles[i][0],particles[i][1]), 3, (0,255,0), -1)
        #cv2.circle(frame, (pos[0],pos[1]), 3, (0,0,255), -1)
        #cv2.imshow('img2',frame)
        #cv2.waitKey(60)

        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,pos[0],pos[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    output.close()

def of_tracker(v, file_name):

    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return
    c,r,w,h = detect_one_face(frame)
    output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2)) 
    frameCounter = frameCounter + 1    

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (20,20),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    old_frame = frame
    old_center = (c,r,w,h)
    mask = np.zeros_like(old_frame)

    c1,c2,c3=0,0,0
    while(1):
    	flag = 0
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

    	c,r,w,h = detect_one_face(frame)

    	if (c+w+r+h) ==0:
    		flag = 1
    		old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    		mask = np.zeros(old_gray.shape, np.uint8)
    		c,r,w,h = old_center
    		mask[r:r+h, c:c+w] = old_gray[r:r+h, c:c+w]
    		p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
    		#print(p0)
    		if p0 is None:
    			c,w,r,h = old_center
    			flag = 0
    		else:
    			p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    			pos = np.mean(p1, axis=0)
    		#print(pos)
    		#for i in range(p0.shape[0]):
    		#	cv2.circle(frame, (int(pos[0,0]),int(pos[0,1])), 3, (255,0,0), -1)
    		#	cv2.circle(frame, (int(p1[i,0,0]),int(p1[i,0,1])), 3, (0,0,255), -1)
        	#	cv2.imshow('img2',frame)
        	#	cv2.waitKey(100) & 0xff

        # test to see if face identified correctly
    	#cv2.rectangle(frame,(c,r),(c+w,r+h),(255,0,0),2)
    	#if flag==0:
    	#	cv2.circle(frame, (c+w/2,r+h/2), 3, (0,255,0), -1)
    	#else:
    	#	cv2.circle(frame, (int(pos[0,0]),int(pos[0,1])), 3, (0,255,0), -1)
    	#cv2.imshow('img',frame)
    	#cv2.waitKey(80) & 0xff
        
    		
    	# write the result to the output file
    	if flag==0:
        	output.write("%d,%d,%d\n" % (frameCounter,c+w/2,r+h/2)) # Write as frame_index,pt_x,pt_y
        elif flag==1:
        	output.write("%d,%d,%d\n" % (frameCounter,int(pos[0,0]),int(pos[0,1])))
        frameCounter = frameCounter + 1
        old_frame = frame
        old_center = (c,r,w,h)

    cv2.destroyAllWindows()
    v.release()

def kalman_tracker(v, file_name):
    # Open output file
    output_name = sys.argv[3] + file_name
    output = open(output_name,"w")

    frameCounter = 0
    # read first frame
    ret ,frame = v.read()
    if ret == False:
        return

    # detect face in first frame
    c,r,w,h = detect_one_face(frame)
    #print(c,r,w,h)

    kalman = cv2.KalmanFilter(4,2,0)
    state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state


    # Write track point for first frame
    pt = (frameCounter,c+w/2,r+h/2)
    output.write("%d,%d,%d\n" % pt) # Write as 0,pt_x,pt_y
    frameCounter = frameCounter + 1

    # set the initial tracking window
    track_window = (c,r,w,h)

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        #print(ret)
        pred = kalman.predict()
        state = (int(pred[0]),int(pred[1]))
        noise = kalman.measurementNoiseCov * np.random.randn(1, 1)
        
        c,r,w,h = detect_one_face(frame)
        #measure = np.array([[91.997997],[67],[92],[66.997]])
        if (c+r+w+h) != 0:
            measure = np.array(np.dot(kalman.measurementMatrix,[c+w/2,r+h/2,0,0]),dtype='float64') 
            state = kalman.correct(measure)
    
        #process_noise = sqrt(kalman.processNoiseCov) * np.random.randn(2, 1)
        #state = np.dot(kalman.transitionMatrix, state) #+ process_noise

        #print(state[0],state[1])
        # Draw it on image
        #img2 = cv2.polylines(frame,[pts],True, 255,2)
        #cv2.circle(frame, (int(state[0]),int(state[1])), 3, (0,0,255), -1)
        #cv2.imshow('img2',frame)
        #cv2.waitKey(100) & 0xff
        
        # write the result to the output file
        output.write("%d,%d,%d\n" % (frameCounter,state[0],state[1])) # Write as frame_index,pt_x,pt_y
        frameCounter = frameCounter + 1

    cv2.destroyAllWindows()
    output.close()

        


if __name__ == '__main__':
    question_number = -1
   
    # Validate the input arguments
    if (len(sys.argv) != 4):
        help_message()
        sys.exit()
    else: 
        question_number = int(sys.argv[1])
        if (question_number > 4 or question_number < 1):
            print("Input parameters out of bound ...")
            sys.exit()

    # read video file
    video = cv2.VideoCapture(sys.argv[2]);

    if (question_number == 1):
        camshift_tracker(video, "output_camshift.txt")
    elif (question_number == 2):
        particle_tracker(video, "output_particle.txt")
    elif (question_number == 3):
        kalman_tracker(video, "output_kalman.txt")
    elif (question_number == 4):
        of_tracker(video, "output_of.txt")

