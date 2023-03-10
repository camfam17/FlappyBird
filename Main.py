import cv2 as cv
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
import time
import random

game_on = True
frame_width = 600
frame_height = 600


class Board():
	
	def __init__(self):
		
		self.bird = Bird()
		self.pipes = PipeController()
		
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		# cv.cvtColor(self.img, cv.COLOR_GRAY2RGB)
		
	
	def tick(self):
		self.bird.tick()
		self.pipes.tick()
	
	
	def render(self):
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		
		
		# self.img = self.bird.render(self.img)
		self.bird.render(self.img)
		self.pipes.render(self.img)
		
		
		cv.imshow('Game', self.img)
		
	
	def flap(self):
		self.bird.flap()


class Bird():
	
	def __init__(self):
		
		self.x = int(frame_width/2)
		self.y = int(frame_height/2)
		
		self.x_vector = -10
		
		self.width = int(0.05 * frame_width)
		self.height = int(0.05 * frame_height)
		print('width, height', self.width, self.height)
		# make a static matrix of the bird & just copy-pase it onto the image at the right x,y location
		self.sprite = np.zeros((self.width, self.height, 3), np.uint8)
		self.sprite[:, :, :] = (0, 255, 0)
		
		pass
	
	def tick(self):
		# The below works well enough for a start
		'''self.x_vector += 1
		a = -0.1
		b = 6
		self.y_vector = 2*a*self.x_vector + b
		self.y_vector = int(self.y_vector)
		
		self.y = self.y - self.y_vector
		'''
		# The below works best so far
		'''self.x_vector += 1
		a = -0.35
		b = 10
		self.y_vector = 2*a*self.x_vector + b
		self.y_vector = int(self.y_vector)
		
		self.y = self.y - self.y_vector
		'''
		
		self.x_vector += 1
		a = -0.35
		b = 10
		self.y_vector = 2*a*self.x_vector + b
		self.y_vector = int(self.y_vector)
		
		self.y = self.y - self.y_vector
		if self.y < -self.height:
			self.y = -self.height
		# print(f'x: {self.x_vector}, y: {self.y_vector}')
		
		# self.check_collision() # TODO: function to check collision with pipes
		
		# TODO when y goes negative, it appears at the top of the screen due to negative index. 
		# Add something to stop this, also killing the player if they go off the bottom of the screen
		
	
	def render(self, img):
		
		# img[self.x : self.x + self.width, self.y : self.y + self.height] = self.sprite
		# img[self.x : self.x + self.width, self.y : self.y + self.height, :] = (0, 255, 0)
		img[self.y : self.y + self.height, self.x : self.x + self.width, :] = (0, 255, 0)
		# img = cv.rectangle(img, (self.x, self.y), (self.x+self.width, self.y+self.height), (0, 255, 0), -1)
	
	def flap(self):
		print('flap')
		self.x_vector = 0


pipe_speed = 3
pipe_width = int(0.08 * frame_width)
gap_height = int(0.25 * frame_height)
class PipeController():
	
	def __init__(self):
		# Pipes will have an X position that changes over time as the pipe moves leftward 
		# Pipes will have a static height (Y position?)
		
		# The game will be continuous (never ending) and thus will require procedural generation of pipes
		n_pipes = 100
		step = 250
		end = n_pipes * step
		self.xs = list(range(frame_width + 100, frame_width + end, step))
		self.ys = []
		for i in range(n_pipes):
			
			self.ys.append(int(frame_height/2) + random.randint(-0.2*frame_height, 0.2*frame_height))
	
	def tick(self):
		# for x in self.xs:
		# 	x -= pipe_speed
		for i in range(len(self.xs)):
			self.xs[i] -= pipe_speed
		
	
	def render(self, img):
		
		for x, y in zip(self.xs, self.ys):
			cv.rectangle(img, (x, y), (x + pipe_width, frame_height), (0, 255, 255), -1)
			cv.rectangle(img, (x, 0), (x + pipe_width, y - gap_height), (0, 255, 255), -1)
		
		


def key_press(key):
	global game_on
	
	print(f'Key Pressed: {key}, type: {type(key)}')
	if type(key) == Key:
		print('yes', key.value)
	if key == Key.space or key == Key.up or key == 'w':
		board.flap()
	elif key == Key.esc or key == 'q': # q not working to quit
		print('quit')
		game_on = False
		# cv.destroyAllWindows()
	elif key == 'g':
		print('press g')

def key_press2(key):
	global game_on
	
	# print(key.char)
	
	try:
		key_code = key.char
	except AttributeError:
		key_code = key.name
	
	print(key_code)
	
	if key_code == 'space' or key_code == 'up' or key_code == 'w':
		board.flap()
	elif key_code == 'esc' or key_code == 'q':
		game_on = False
	
	
	pass

def key_release(key):
	# print(f'Key Released: {key}')
	pass


fps = 60
frame = 1/fps
if __name__ == '__main__':
	
	board = Board()
	
	listener = keyboard.Listener(on_press=key_press2, on_release=key_release)
	listener.start()
	
	count = 0
	while game_on:
		
		start_time = time.time()
		
		
		#### Execution ####
		board.tick()
		board.render()
		
		count += 1
		if count % 60 == 0: print('Count:', count)
		
		
		cv_key = cv.waitKey(1)
		if cv_key == 27 or cv_key == ord('q') or cv.getWindowProperty('Game', cv.WND_PROP_VISIBLE) < 1: # Press esc or q to quit
			# cv.destroyAllWindows()
			break
		
		delta_time = time.time() - start_time
		sleep_time = frame - delta_time
		if sleep_time > 0:
			time.sleep(sleep_time)
	
	cv.destroyAllWindows()