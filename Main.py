import cv2 as cv
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
import time

game_on = True
frame_width = 600
frame_height = 600

class Board():
	
	def __init__(self):
		
		self.bird = Bird()
		self.pipes = PipeController()
		
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		# cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
		
		pass
		
	
	def tick(self):
		self.bird.tick()
		self.pipes.render()
		pass
	
	
	def render(self):
		# self.img = self.bird.render(self.img)
		self.bird.render(self.img)
		self.pipes.render()
		
		
		cv.imshow('Game', self.img)
		


class Bird():
	
	def __init__(self):
		
		self.x = int(frame_width/2)
		self.y = int(frame_height/2)
		
		self.width = int(0.05 * frame_width)
		self.height = int(0.05 * frame_height)
		print('width, height', self.width, self.height)
		# make a static matrix of the bird & just copy-pase it onto the image at the right x,y location
		self.sprite = np.zeros((self.width, self.height, 3), np.uint8)
		self.sprite[:, :, :] = (0, 255, 0)
		
		pass
	
	def tick(self):
		
		pass
	
	def render(self, img):
		
		# img[self.x : self.x + self.width, self.y : self.y + self.height] = self.sprite
		img[self.x : self.x + self.width, self.y : self.y + self.height, :] = (0, 255, 0)
		# img = cv.rectangle(img, (self.x, self.y), (self.x+self.width, self.y+self.height), (0, 255, 0), -1)
		


class PipeController():
	
	def __init__(self):
		# Pipes will have an X position that changes over time as the pipe moves leftward 
		# Pipes will have a static height (Y position?)
		pipes = []
	
	def tick(self):
		
		pass
	
	def render(self):
		
		pass


def key_press(key):
	
	global game_on
	
	# print(f'Key Pressed: {key}')
	if key == Key.space or key == Key.up or key == 'w':
		print('space')
	elif key == Key.esc or key == 'q':
		print('esc')
		game_on = False
		cv.destroyAllWindows()

def key_release(key):
	print(f'Key Released: {key}')


fps = 60
frame = 1/fps
if __name__ == '__main__':
	
	board = Board()
	
	listener = keyboard.Listener(on_press=key_press, on_release=key_release)
	listener.start()
	
	count = 0
	while game_on:
		
		start_time = time.time()
		
		
		#### Execution ####
		board.tick()
		board.render()
		
		
		cv_key = cv.waitKey(1)
		if cv_key == 27 or cv_key == ord('q'): # Press esc or q to quit
			cv.destroyAllWindows()
			break
		
		delta_time = time.time() - start_time
		sleep_time = frame - delta_time
		if sleep_time > 0:
			time.sleep(sleep_time)