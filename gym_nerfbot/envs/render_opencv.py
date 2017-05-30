import numpy as np
import cv2


class SimViewer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.img = np.zeros((height,width,3), np.uint8)
        self.target_history = []

    def render(self, mode='rgb_array', history=False):
        if history==True:
            self.draw_history()
        if mode=='rgb_array':
            return self.img

    def draw_history(self):
        
        
# def main():
#     cap = cv2.VideoCapture(0)

#     while(True):
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Our operations on the frame come here
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Display the resulting frame
#         cv2.imshow('frame',gray)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()


def main():
    s = SimViewer(640,480)
    
    while(True):
        cv2.imshow('img',s.img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
