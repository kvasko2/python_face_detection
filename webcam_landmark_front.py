import cv2
import dlib
import numpy

cap = cv2.VideoCapture(0)
PREDICTOR_PATH = "./features/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='./features/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)

img_majora = cv2.imread("./overlays/mask.png")
# Create ROI (not really sure what that is yet).
rows, cols, channels = img_majora.shape

# Create masks of logo and its inverse.
majora_mask_gray = cv2.cvtColor(img_majora, cv2.COLOR_BGR2GRAY)
ret, orig_majora_mask = cv2.threshold(majora_mask_gray, 10, 255, cv2.THRESH_BINARY)
orig_majora_mask_inv = cv2.bitwise_not(orig_majora_mask)



def get_landmarks(im):
    rects = cascade.detectMultiScale(
        im,
        1.3,
        5,
        minSize=(50,50)
    )
    if rects != ():
        #print "rects: ", rects
        x,y,w,h =rects[0]
        #print "x: ", x, ", y: ", y, ", w: ", w, ", h: ", h
        rect=dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    else:
        x,y,w,h = [0,0,0,0]
        rect=dlib.rectangle(long(x),long(y),long(x+w),long(y+h))
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def add_overlay(im):
    faces = cascade.detectMultiScale(
        im,
        1.3,
        5,
        minSize=(50,50)
    )

    for (fx, fy, fw, fh) in faces:
        # Debugging frame to show that face is found.
        #cv2.rectangle(im, (fx, fy), (fx+fw, fy+fh), (0,0,255), 2)

        overlay_width = int(fw * 1.5)
        overlay_height = int(fh * 1.5)

        x = fx - ((overlay_width - fw) / 2)
        y = fy - ((overlay_height - fh) / 2)

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        #print ("x: ", x, ", y: ", y, ", width: ", overlay_width, ", height: ", overlay_height)

        roi = im[y:y+overlay_height, x:x+overlay_width]

        # Resize the image and masks.
        mm = cv2.resize(img_majora, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
        majora_mask = cv2.resize(orig_majora_mask, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
        majora_mask_inv = cv2.resize(orig_majora_mask_inv, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)

        # Separate the background and foreground (the video feed and the overlaid image).
        bg = cv2.bitwise_and(roi, roi, mask=majora_mask_inv)
        fg = cv2.bitwise_and(mm, mm, mask=majora_mask)

        # Place the overlay on the ROI and modify the main image.
        dst = cv2.add(bg, fg)
        im[y:y+overlay_height, x:x+overlay_width] = dst

        break
    # end for
# end add_overlay

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

try:
    while (True):
        _, im=cap.read()
        landmarks = get_landmarks(im)
        add_overlay(im)
        cv2.imshow('Result',annotate_landmarks(im,landmarks))
        cv2.waitKey(1)
except KeyboardInterrupt:
    print "Exiting because of ctrl+c"
    pass

cv2.destroyAllWindows()
