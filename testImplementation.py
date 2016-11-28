from eventBasedAnimationClass import EventBasedAnimationClass
from Tkinter import *
from GesturesApi import GestureProcessor
from PIL import Image, ImageTk
# Import statements from: 
# http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
HEIGHT = 1366
WIDTH = 768

# Subclasses Object from here:
# http://www.cs.cmu.edu/~112/notes/eventBasedAnimationClass.py
class GestureDemo(EventBasedAnimationClass):
    def __init__(self):
    	self.write_path_to_file = False
        self.gp = GestureProcessor("Gesture_data.txt")  # default to usual file
        self.width = HEIGHT
        self.height = WIDTH
        super(GestureDemo, self).__init__(width=self.width, height=self.height)
        self.timerDelay = 1000 / 30 # 30 FPS
        self.bindGestures()
        self.CVHandles = []
        self.bgHandle = None
        self.trackCenter = False
        self.trail = False

    def bindGestures(self):
        self.gp.bind("Infinity", lambda: self.drawLukas())
        self.gp.bind("Diagonal Bottom Left to Top Right",
                     lambda: self.drawSmiley())

    def bindHandlers(self):
        self.root.bind("<KeyPress>", lambda event: self.onKeyDown(event))
        self.root.bind("<KeyRelease>", lambda event: self.onKeyUp(event))

    def onMousePressed(self, event):
        print "Mouse Clicked at:", (event.x, event.y)

    def onKeyPressed(self, event):
        if event.char == 'l':
            self.write_path_to_file = not self.write_path_to_file
            if self.write_path_to_file:
        	    self.path_file = open('test123.txt','w')
            else:
        		close(self.path_file)
        elif event.char == 'r':
            self.gp.saveNext()
        elif event.char == 's':
            self.trackCenter = not self.trackCenter
        elif event.char == 'c':
            self.trail = not self.trail
        elif event.char == 'd':
            self.canvas.delete(ALL)
            self.drawBG()
        elif event.char == 'b':
            self.bindGestures()
        elif event.char == 'q':
            self.onClose()
            exit()

    def onTimerFired(self):
        self.gp.process()

    # OpenCV Image drawing adapted from:
    # http://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
    def drawCVImages(self):
        for handle in self.CVHandles:
            self.canvas.delete(handle)
        self.CVHandles = []

        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.original,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.gp.draw()
        cv2image = GestureProcessor.getRGBAFromBGR(self.gp.drawingCanvas,
                                                   self.width / 2,
                                                   self.height / 2)
        self.imagetk2 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        cv2image = GestureProcessor.getRGBAFromGray(self.gp.thresholded,
                                                    self.width / 2,
                                                    self.height / 2)
        self.imagetk3 = ImageTk.PhotoImage(image=Image.fromarray(cv2image))

        self.CVHandles.append(self.canvas.create_image(0, 0, image=self.imagetk,
                              anchor="nw"))
        self.CVHandles.append(self.canvas.create_image(HEIGHT, WIDTH,
                              image=self.imagetk2, anchor="se"))
        self.CVHandles.append(self.canvas.create_image(0, WIDTH,
                              image=self.imagetk3, anchor="sw"))

        self.CVHandles.append(self.canvas.create_text(HEIGHT, 0,
                              text=self.gp.lastAction, anchor="ne",
                              font="15"))
        self.CVHandles.append(self.canvas.create_text(HEIGHT, 20,
                              text="Distance: " + str(round(
                                                      self.gp.handDistance, 3)),
                              anchor="ne", font="15"))
        self.CVHandles.append(self.canvas.create_text(HEIGHT, 40,
                              text=str(self.gp.getScaledCenter()),
                              anchor="ne", font="15"))
        if self.write_path_to_file:
            print("writing...")
            center = self.gp.getScaledCenter()
            x,y = 1000*center[0],1000*(1-center[1])
            self.path_file.write('{},{}\n'.format(y,x))

    def drawBG(self):
        self.bgHandle = self.canvas.create_rectangle(self.width/2, 0,
                                                     self.width, self.height/2,
                                                     fill="white")

    def redrawAll(self):
        self.drawCVImages()

    def run(self):
        super(GestureDemo, self).run()
        self.onClose()

    def onClose(self):
        self.gp.close()  # MUST DO THIS

GestureDemo().run()
