import cv2 as cv
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
import time
import random

game_on = True
game_state = 'play'
frame_width = 600
frame_height = 600


class Board():
	
	def __init__(self):
		
		self.bird = Bird()
		self.pipes = PipeController()
		
		self.img = np.zeros(shape=(frame_width, frame_height, 3))		
	
	def tick(self):
		self.bird.tick(self.img)
		self.pipes.tick()
	
	
	def render(self):
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		
		
		self.bird.render(self.img)
		self.pipes.render(self.img)
		
		
		if game_state == 'pause':
			(text_width, text_height), _ = cv.getTextSize('PAUSED', fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.1, thickness=5)
			cv.putText(self.img, 'PAUSED', org=(int(frame_width/2 - text_width/2), int(frame_height/2 - text_height/2)), 
	      				fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.1, color=(255, 255, 255), thickness=5)
		elif game_state == 'dead':
			(text_width1, text_height1), _ = cv.getTextSize('YOU DIED', fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.1, thickness=5)
			cv.putText(self.img, 'YOU DIED', org=(int(frame_width/2 - text_width1/2), int(frame_height/2 - text_height1/2)), 
	      				fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.1, color=(255, 255, 255), thickness=5)
			
			(text_width2, text_height2), _ = cv.getTextSize('Press any key to continue', fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
			cv.putText(self.img, 'Press any key to continue', org=(int(frame_width/2 - text_width2/2), int(frame_height/2 + text_height2/2)), 
	      				fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)
			
		
		# cv.line(self.img, (int(frame_width/2), 0), (int(frame_width/2), frame_height), color=(0, 0, 255), thickness=1)
		# cv.line(self.img, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), color=(0, 0, 255), thickness=1)
		
		cv.imshow('Game', self.img)
		
	
	def flap(self):
		self.bird.flap()
	
	def restart(self):
		global game_state
		
		print('restart')
		self.__init__()
		game_state = 'play'


class Bird():
	
	def __init__(self):
		
		self.x_vector = -10
		
		self.width = int(0.05 * frame_width)
		self.height = int(0.05 * frame_height)
		
		self.x = int(frame_width/2 - self.width/2)
		self.y = int(frame_height/2 - self.height/2)
		
		print('width, height', self.width, self.height)
		# make a static matrix of the bird & just copy-pase it onto the image at the right x,y location
		self.sprite = np.zeros((self.width, self.height, 3), np.uint8)
		self.sprite[:, :, :] = (0, 255, 0)
		
		pass
	
	def tick(self, img):
		global game_state
		
		self.check_collision(img)
		
		self.x_vector += 1
		a = -0.35
		b = 10
		self.y_vector = 2*a*self.x_vector + b # derivative of parabola ax^2 + bx
		self.y_vector = int(self.y_vector)
		
		self.y = self.y - self.y_vector
		if self.y < -self.height: # block the bird from flying up indefinitely
			self.y = -self.height
		
		if self.y > frame_height - self.height:
			self.y = frame_height - self.height + 2
			game_state = 'dead'
		
		
		# self.check_collision() # TODO: function to check collision with pipes
	
	def check_collision(self, img):
		global game_state
		# check collision
		# get a one pixel thick square around the bird. If any of those pixels is the colour of the pipe, then collision has happened
		
		top_line = img[self.y-1, self.x-1 : self.x + self.width + 1, :]
		bottom_line = img[self.y + self.height, self.x-1 : self.x + self.width + 1, :]
		left_line = img[self.y-1 : self.y + self.height + 1, self.x-1, :] 
		right_line = img[self.y-1 : self.y + self.height + 1, self.x + self.width, :]
		
		# if pipe_colour in [*top_line, *bottom_line, *left_line, *right_line]:
		# if (list(pipe_colour) - [*top_line, *bottom_line, *left_line, *right_line]).all():
		# colours = set([*top_line, *bottom_line, *left_line, *right_line])
		# if pipe_colour in colours:
		# 	print('COLLIDE!!!!!')
		
		# colours = map( lambda x: cv.cvtColor(x, cv.COLOR_BGR2HSV), [*top_line, *bottom_line, *left_line, *right_line])
		colours = [*top_line, *bottom_line, *left_line, *right_line]
		colours = set(map(self.rgb_to_string, colours))
		
		pipe_string = self.rgb_to_string(pipe_colour)
		
		print(colours, 'cwqccqw', pipe_string)
		if pipe_string in colours:
			print('COLLIDE!!!!!')
			game_state = 'dead'
		
		
		pass
	
	def rgb_to_string(self, *args):
		
		args = args[0]
		
		string = ''
		string += str(int(args[0]))
		string += str(int(args[1]))
		string += str(int(args[2]))
		
		return string
		
		pass
	
	def format_colour_string(self, colour):
		return str(list(colour)).replace(',', '.').replace('[', '').replace(']', '').replace(' ', '')
	
	def render(self, img):
		
		# img[self.x : self.x + self.width, self.y : self.y + self.height] = self.sprite
		# img[self.x : self.x + self.width, self.y : self.y + self.height, :] = (0, 255, 0)
		img[self.y : self.y + self.height, self.x : self.x + self.width, :] = (0, 255, 0)
		# img = cv.rectangle(img, (self.x, self.y), (self.x+self.width, self.y+self.height), (0, 255, 0), -1)
		
		
		# img[self.y-1, self.x-1 : self.x + self.width + 1, :] = (0, 255, 255) # top line
		# img[self.y + self.height, self.x-1 : self.x + self.width + 1, :] = (255, 255, 255) # bottom line
		# img[self.y-1 : self.y + self.height + 1, self.x-1, :] = (255, 255, 255) # left line
		# img[self.y-1 : self.y + self.height + 1, self.x + self.width, :] = (255, 255, 255) # right line
	
	def flap(self):
		print('flap')
		self.x_vector = 0


pipe_speed = 3
pipe_width = int(0.08 * frame_width)
gap_height = int(0.25 * frame_height)
pipe_colour = [0, 255, 255]
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
		for i in range(n_pipes): # top height is good, find better bottom height (lower it slightly)
			self.ys.append(int(frame_height/2) + random.randint(-0.2*frame_height, 0.2*frame_height + gap_height))
		
		# the x and y coordinates of eacch pipe refers to the top left corner of the bottom pipe
	
	def tick(self):
		# for x in self.xs:
		# 	x -= pipe_speed
		for i in range(len(self.xs)):
			self.xs[i] -= pipe_speed
		
	
	def render(self, img):
		
		for x, y in zip(self.xs, self.ys):
			cv.rectangle(img, (x, y), (x + pipe_width, frame_height), pipe_colour, -1)
			cv.rectangle(img, (x, 0), (x + pipe_width, y - gap_height), pipe_colour, -1)
			cv.circle(img, (x, y), 3, (0, 0, 255), -1)
		
		


def key_press(key):
	global game_on
	global game_state
	
	# print(key.char)
	
	try:
		key_code = key.char
	except AttributeError:
		key_code = key.name
	
	print(key_code)
	
	if game_state == 'dead':
		# restart game
		board.restart()
		pass
	else:
		if key_code == 'space' or key_code == 'up' or key_code == 'w':
			board.flap()
		elif key_code == 'esc' or key_code == 'q':
			game_on = False
		elif key_code == 'p':
			if game_state == 'play':
				game_state = 'pause'
			elif game_state == 'pause':
				game_state = 'play'

def key_release(key):
	# print(f'Key Released: {key}')
	pass


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
		if game_state == 'play':
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