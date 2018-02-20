import cv2
import numpy as np 

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve
color = 'FR' # FR means foreground & BB means background
pts_f = []
pts_b = []

# mouse callback function
def mouse_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, pts_f, pts_b

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
        pts_f.append((current_former_x, current_former_y))
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
                if color == 'FR':
                    pts_f.append((current_former_x, current_former_y))
                elif color == 'BB':
                    pts_b.append((current_former_x, current_former_y))
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
            if color == 'FR':
                pts_f.append((current_former_x, current_former_y))
            elif color == 'BB':
                pts_b.append((current_former_x, current_former_y))
            #print(current_former_x, current_former_y)

    return former_x,former_y    



im = cv2.imread("astronaut.png")
cv2.namedWindow("CV HW4 - Bhavya Ghai")
cv2.setMouseCallback('CV HW4 - Bhavya Ghai',mouse_draw)
print("Foreground Selected")

marking = np.zeros((im.shape[1],im.shape[0],3), np.uint8)
marking.fill(255)
while(1):
    cv2.imshow('CV HW4 - Bhavya Ghai',im)
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
        im = cv2.imread("astronaut.png")
        print('All lines cleared')
        pts_b = []
        pts_f = []
    elif k==13:
        #for i in pts_f:
        #    marking[i[1],i[0]] = (0,0,255)
        #for i in pts_b:
        #    marking[i[1],i[0]] = (255,0,0)
        orig = cv2.imread("astronaut.png")
        ind = np.array(np.where(im!=orig)).T
        c,c1 = 0,0
        print(ind.shape)
        for i in ind:
            x,y = i[0],i[1]
            if list(im[x,y])== [0,0,255]:
                marking[x,y] = [0,0,255]
                c = c+1
            elif list(im[x,y])== [255,0,0]:
                marking[x,y] = [255,0,0]
                c = c+1
            else:
                print orig[x,y],"      ",im[x,y] 
                c1 = c1+1
                #print im[x,y]
        #marking = marking + (im-orig)
        #print ind.shape[0],c,c1
        #cv2.imshow('marking', marking)
        #cv2.waitKey(0)
        #print("Show segemented Output")

cv2.destroyAllWindows()
#print "Foreground \n",pts_f
#print "Background \n",pts_b
#for i in pts_f:
#    print marking[i[1],i[0]]