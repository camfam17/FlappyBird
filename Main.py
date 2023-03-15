import cv2 as cv
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
import time
import random

game_on = True
game_state = 'play'
frame_width = frame_height = 600

class Board():
	
	def __init__(self, nbirds=1):
		
		# self.bird = Bird()
		self.nbirds = nbirds
		self.birds= []
		for i in range(nbirds):
			self.birds.append(Bird())
		self.pipes = PipeController()
		
		self.points = 0
		
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		self.start_time = time.time()
		self.fps = 0
	
	def tick(self):
		
		# Measure actual frames per second
		prev_time = self.start_time
		self.start_time = time.time()
		delta_time = self.start_time - prev_time
		if delta_time > 0:
			if not abs(self.fps - int(1 / delta_time)) < 10:
				self.fps = int(1 / delta_time)
		
		# self.bird.tick(self.img)
		for bird in self.birds:
			bird.tick(self.img)
		self.pipes.tick()
		
		for bird in self.birds:
			self.check_points(bird)
		
	
	def render(self):
		self.img = np.zeros(shape=(frame_width, frame_height, 3))
		
		
		# self.bird.render(self.img)
		for bird in self.birds:
			bird.render(self.img)
		self.pipes.render(self.img)
		
		# TODO make font scale & thickness with frame_width & frame_height
		if game_state == 'pause':
			(text_width, text_height), _ = cv.getTextSize('PAUSED', fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.8333e-3*frame_width, thickness=int(8.333e-3*frame_width))
			cv.putText(self.img, 'PAUSED', org=(int(frame_width/2 - text_width/2), int(frame_height/2 - text_height/2)), 
	      				fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.8333e-3*frame_width, color=(255, 0, 255), thickness=int(8.333e-3*frame_width))
		elif game_state == 'dead':
			(text_width1, text_height1), _ = cv.getTextSize('YOU DIED', fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.8333e-3*frame_width, thickness=int(8.333e-3*frame_width))
			cv.putText(self.img, 'YOU DIED', org=(int(frame_width/2 - text_width1/2), int(frame_height/2 - text_height1/2)), 
	      				fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.8333e-3*frame_width, color=(255, 0, 255), thickness=int(8.333e-3*frame_width))
			
			(text_width2, text_height2), _ = cv.getTextSize('Press r to continue', fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=8.333e-4*frame_width, thickness=int(3.333e-3*frame_width))
			cv.putText(self.img, 'Press r to continue', org=(int(frame_width/2 - text_width2/2), int(frame_height/2 + text_height2/2)), 
	      				fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=8.333e-4*frame_width, color=(255, 0, 255), thickness=int(3.333e-3*frame_width))
			
		
		(_, points_height), _ = cv.getTextSize('Points: ' + str(self.points), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.333e-3*frame_width, thickness=int(3.333e-3*frame_width))
		cv.putText(self.img, 'Points: ' + str(self.points), org=(10, points_height + 10), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
	     			fontScale=1.333e-3*frame_width, color=(0, 0, 255), thickness=int(3.333e-3*frame_width))
		
		(_, fps_height), _ = cv.getTextSize(str(self.fps) + 'fps', fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)
		cv.putText(self.img, str(self.fps) + 'fps', org=(10, fps_height + points_height + int(0.05*frame_width)), 
	     		    fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.333e-3*frame_width, color=(0, 0, 255), thickness=int(3.333e-3*frame_width))
		
		# cv.line(self.img, (int(frame_width/2), 0), (int(frame_width/2), frame_height), color=(0, 0, 255), thickness=1)
		# cv.line(self.img, (0, int(frame_height/2)), (frame_width, int(frame_height/2)), color=(0, 0, 255), thickness=1)
		
		# self.img = cv.resize(src=self.img, dsize=(300, 300))
		# self.next_pipe_coords()
		cv.imshow('Flappy Bird', self.img)
	
	
	def check_points(self, bird):
		
		bird_x1 = bird.x
		bird_x2 = bird_x1 + bird.width
		
		xs = self.pipes.xs
		
		for i, x in enumerate(xs):
			if x in range(bird_x1, bird_x2) and self.pipes.pointed[i] == False:
				print('POINT!!!!!')
				self.points += 1
				self.pipes.pointed[i] = True
	
	
	def flap(self, birdID):
		# self.bird.flap()
		if birdID < 0:
			for bird in self.birds:
				bird.flap()
		else:
			self.birds[birdID].flap()
	
	def restart(self):
		global game_state
		
		print('restart')
		self.__init__(self.nbirds)
		game_state = 'play'
	
	def pipe_coords(self):
		
		pipe_coords = []
		for i in range(3):
			pipe_coords.append(list((self.pipes.xs[i], self.pipes.ys[i])))
		# print(pipe_coords)
		
		return pipe_coords
	
	def next_pipe_coords(self):
		
		for x, y in zip(self.pipes.xs, self.pipes.ys):
			if x > self.birds[0].x:
				
				# cv.circle(self.img, (int(x), y), 4, (0, 0, 255), -1)
				return [x, y]
	
	def bird_coords(self):
		return [self.birds[0].x, self.birds[0].y]


class Bird():
	
	def __init__(self):
		
		self.x_vector = -10 + random.randint(-2, 2)
		
		self.width = int(0.05 * frame_width)
		self.height = int(0.05 * frame_height)
		
		self.x = int(frame_width/2 - self.width/2)
		self.y = int(frame_height/2 - self.height/2)
		
		self.sprite = np.zeros((self.width, self.height, 3), np.uint8)
		self.sprite[:, :, :] = (0, 255, 0)
		
	
	def tick(self, img):
		global game_state
		
		if self.y > frame_height - self.height -1 : # kill the bird if it falls to the bottom of the screen
			self.y = frame_height - self.height + 2
			game_state = 'dead'
			return
		
		collision = self.check_collision(img)
		if collision:
			return
		
		self.x_vector += 1
		a = -5.8333e-4 * frame_height # a = -0.35
		b = 0.016667 * frame_height # b = 10
		self.y_vector = 2*a*self.x_vector + b # derivative of parabola ax^2 + bx
		self.y_vector = int(self.y_vector)
		# Maybe have a terminal velocity i.e. self.y_vector can get bigger only up to a certain point
		
		self.y = self.y - self.y_vector
		if self.y < -self.height: # block the bird from flying up indefinitely
			self.y = -self.height
			game_state = 'dead'
			return
	
	
	def render(self, img):
		
		img[self.y : self.y + self.height, self.x : self.x + self.width, :] = (0, 255, 0)
		# img = cv.rectangle(img, (self.x, self.y), (self.x+self.width, self.y+self.height), (0, 255, 0), -1)
		
		# img[self.y-1, self.x-1 : self.x + self.width + 1, :] = (255, 255, 255) # top line
		# img[self.y + self.height, self.x-1 : self.x + self.width + 1, :] = (255, 255, 255) # bottom line
		# img[self.y-1 : self.y + self.height + 1, self.x-1, :] = (255, 255, 255) # left line
		# img[self.y-1 : self.y + self.height + 1, self.x + self.width, :] = (255, 255, 255) # right line
	
	
	def check_collision(self, img):
		global game_state
		
		top_line = img[self.y-1, self.x-1 : self.x + self.width + 1, :]
		bottom_line = img[self.y + self.height, self.x-1 : self.x + self.width + 1, :]
		left_line = img[self.y-1 : self.y + self.height + 1, self.x-1, :] 
		right_line = img[self.y-1 : self.y + self.height + 1, self.x + self.width, :]
		
		colours = [*top_line, *bottom_line, *left_line, *right_line]
		colours = set(map(self.rgb_to_string, colours))
		pipe_string = self.rgb_to_string(pipe_colour)
		if pipe_string in colours:
			# print('COLLIDE!!!!!')
			game_state = 'dead'
			return True
		
		return False
		
	
	def rgb_to_string(self, *args):
		
		args = args[0]
		
		string = str(list(map(int, args)))
		
		return string
	
	
	def flap(self):
		self.x_vector = 0


pipe_speed = 0.005 * frame_width # pipe_speed = 3
pipe_width = int(0.08 * frame_width)
gap_height = int(0.25 * frame_height)
pipe_colour = [0, 255, 255]
class PipeController():
	
	def __init__(self):
		# n_pipes = 5
		self.step = int(0.416667 * frame_width) # step = 250
		# end = n_pipes * self.step
		# self.xs = list(range(int(1.166667 * frame_width), frame_width + end, self.step))
		# self.ys = []
		# for i in range(n_pipes): # top height is good, find better bottom height (lower it slightly)
		# 	self.ys.append(int(frame_height/2) + random.randint(-0.2*frame_height, 0.2*frame_height + gap_height))
		
		self.pointed = [False] #* n_pipes
		
		self.xs = [frame_width]
		self.ys = [int(frame_height/2) + random.randint(-0.2*frame_height, 0.2*frame_height + gap_height)]
		for i in range(4):
			self.add_new_pipe()
		
		# the x and y coordinates of eacch pipe refers to the top right corner of the bottom pipe
	
	def tick(self):
		for i in range(len(self.xs)):
			self.xs[i] -= pipe_speed
			
			if self.xs[i] < 0:
				self.xs.pop(i)
				self.ys.pop(i)
				self.pointed.pop(i)
				self.add_new_pipe()
	
	
	def render(self, img):
		
		for x, y in zip(self.xs, self.ys):
			
			x = int(x)
			y = int(y)
			
			cv.rectangle(img, (x, y), (x - pipe_width, frame_height), pipe_colour, -1)
			cv.rectangle(img, (x, 0), (x - pipe_width, y - gap_height), pipe_colour, -1)
			# cv.circle(img, (x, y), 3, (0, 0, 255), -1)
	
	
	def add_new_pipe(self):
		
		# self.xs.append(int(1.166667 * frame_width))
		self.xs.append(self.xs[-1] + self.step)
		self.ys.append(int(frame_height/2) + random.randint(-0.2*frame_height, 0.2*frame_height + gap_height))
		self.pointed.append(False)


def key_press(key):
	global game_on
	global game_state
	
	try:
		key_code = key.char
	except AttributeError:
		key_code = key.name
	
	# print(key_code)
	
	if game_state == 'dead': # press any key to restart
		if key_code == 'r':
			board.restart()
	else:
		if key_code == 'space' or key_code == 'up' or key_code == 'w': # to jump/flap
			board.flap(-1)
		elif key_code == 'esc' or key_code == 'q': # to quit the game
			game_on = False
		elif key_code == 'p': # to pause and unpause
			game_state = 'pause' if game_state == 'play' else 'play'

def key_release(key):
	# print(f'Key Released: {key}')
	pass


fps = 60
frame = 1/fps


class Controller():
	
	def __init__(self, agent):
		
		self.agent = agent
		self.start_loop()
	
	
	def start_loop(self):
		global board
		board = Board(nbirds=1)
		
		listener = keyboard.Listener(on_press=key_press, on_release=key_release)
		if self.agent is None:
			listener.start()
		
		count = 0
		while game_on:
			
			start_time = time.time()
			
			if game_state  == 'dead' and self.agent is not None:
				# process.terminate()
				break
			
			#### Execution ####
			if game_state == 'play':
				board.tick()
			board.render()
			
			if self.agent is not None:
				# jump = self.agent.predict(board.bird_coords(), board.pipe_coords(), board.points, game_state=='dead')
				jump = self.agent.predict(board.points, game_state=='dead', board.bird_coords(), board.next_pipe_coords())
				if jump:
					key_press(Key.space)
			
			
			count += 1
			# if count % fps == 0: print('Count:', count, time.time())
			
			
			cv_key = cv.waitKey(1)
			# if cv_key == 27 or cv_key == ord('q') or cv.getWindowProperty('Flappy Bird', cv.WND_PROP_VISIBLE) < 1: # Press esc or q to quit
			# 	break
			
			delta_time = time.time() - start_time
			sleep_time = frame - delta_time
			if sleep_time > 0:
				time.sleep(sleep_time)
		
		print('Game Exit')
		cv.destroyAllWindows()
	
if __name__ == '__main__':
	
	board = Board(nbirds=1)
	c = Controller(None)
	
	# board = Board(nbirds=1)
	
	# listener = keyboard.Listener(on_press=key_press, on_release=key_release)
	# listener.start()
	
	# count = 0
	# while game_on:
		
	# 	start_time = time.time()
		
		
	# 	#### Execution ####
	# 	if game_state == 'play':
	# 		board.tick()
	# 	board.render()
		
		
	# 	count += 1
	# 	if count % fps == 0: print('Count:', count, time.time())
		
		
	# 	cv_key = cv.waitKey(1)
	# 	if cv_key == 27 or cv_key == ord('q') or cv.getWindowProperty('Flappy Bird', cv.WND_PROP_VISIBLE) < 1: # Press esc or q to quit
	# 		break
		
	# 	delta_time = time.time() - start_time
	# 	sleep_time = frame - delta_time
	# 	if sleep_time > 0:
	# 		time.sleep(sleep_time)
	
	# cv.destroyAllWindows()