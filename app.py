import pygame, sys
import numpy as np
import cv2
from keras.models import load_model

# --- Settings ---
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PREDICT = True

# Load model
MODEL = load_model("trained_model.h5")

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# --- Initialize pygame ---
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 20)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition Board - Fixed")
DISPLAYSURF.fill(BLACK)

iswriting = False
x_coords = []
y_coords = []

def preprocess_for_mnist(roi_gray):
    """
    roi_gray: grayscale numpy array of the drawn area (H x W)
    returns: 28x28 float32 array normalized like MNIST (white digit on black bg)
    """
    if roi_gray.size == 0:
        return None

    # threshold to clean up
    _, th = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep only largest connected component (helps remove stray dots)
    contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # find bounding box for the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    digit = th[y:y+h, x:x+w]

    # resize preserving aspect ratio into 20x20 box (like MNIST preprocessing)
    h_d, w_d = digit.shape
    if h_d > w_d:
        new_h = 20
        new_w = int(round((w_d * 20) / h_d))
    else:
        new_w = 20
        new_h = int(round((h_d * 20) / w_d))
    if new_w == 0 or new_h == 0:
        return None
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad to 28x28 and center using center of mass
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized

    # compute center of mass and shift to center (optional but improves)
    cy, cx = ndimage_center_of_mass(canvas)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    canvas = cv2.warpAffine(canvas, M, (28,28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # invert so digit is white (255) and background black (0)
    canvas = cv2.bitwise_not(canvas)

    # normalize to [0,1]
    canvas = canvas.astype('float32') / 255.0
    return canvas

def ndimage_center_of_mass(img):
    # compute center of mass for a binary image
    # fallback if empty
    try:
        moments = cv2.moments(img)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            # default center
            cy, cx = img.shape[0]//2, img.shape[1]//2
    except Exception:
        cy, cx = img.shape[0]//2, img.shape[1]//2
    return cy, cx

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEMOTION and iswriting:
            x,y = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x,y), 12, 0)  # thicker stroke
            x_coords.append(x); y_coords.append(y)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True
            x_coords = []; y_coords = []

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue

            # bounding box with padding
            xmin = max(min(x_coords) - BOUNDRYINC, 0)
            xmax = min(max(x_coords) + BOUNDRYINC, WINDOWSIZEX)
            ymin = max(min(y_coords) - BOUNDRYINC, 0)
            ymax = min(max(y_coords) + BOUNDRYINC, WINDOWSIZEY)
            x_coords = []; y_coords = []

            # capture entire screen as array (W,H,3) -> convert to H,W,3 for OpenCV:
            surf_arr = pygame.surfarray.array3d(DISPLAYSURF)  # shape (W, H, 3)
            surf_arr = np.transpose(surf_arr, (1, 0, 2))     # to (H, W, 3)
            gray = cv2.cvtColor(surf_arr, cv2.COLOR_RGB2GRAY)

            roi = gray[ymin:ymax, xmin:xmax]
            if roi.size == 0:
                continue

            # preprocess to 28x28 MNIST-like image
            canvas = preprocess_for_mnist(roi)
            if canvas is None:
                continue

            # predict
            x_input = np.reshape(canvas, (1, 28, 28, 1))
            pred = MODEL.predict(x_input)
            digit = np.argmax(pred)
            label = LABELS.get(digit, str(digit))

            # draw rectangle and label
            cv2_rect_left = xmin; cv2_rect_top = ymin
            pygame.draw.rect(DISPLAYSURF, RED, (xmin, ymin, xmax-xmin, ymax-ymin), 2)
            text_surf = FONT.render(label, True, RED, BLACK)
            text_rect = text_surf.get_rect()
            text_rect.left, text_rect.bottom = xmin, max(ymin-5, 20)
            DISPLAYSURF.blit(text_surf, text_rect)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:  # clear screen
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
