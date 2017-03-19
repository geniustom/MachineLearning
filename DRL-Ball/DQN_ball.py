import cv2
import numpy as np
import matplotlib.pyplot as plt
from BrainDQN_Nature import BrainDQN


##################################################################################################################
##################################################################################################################
import pygame

BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)

GAME_WIDTH = 300
GAME_HEIGHT = 400
SCREEN_SIZE = [GAME_WIDTH,GAME_HEIGHT]
BAR_SIZE = [100, 10]
BALL_SIZE = [15, 15]
BALL_SPEED = 10

# 神經網路的輸出
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]
ALIVE_REWARD = 0    #存活獎勵
WIN_REWARD = 1    #獎勵
LOSE_REWARD = -1  #懲罰
 
class Game(object):
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(SCREEN_SIZE)
		pygame.display.set_caption('打磚塊')
 
		self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
		self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
 
		self.ball_dir_x = -1 # -1 = left 1 = right  
		self.ball_dir_y = -1 # -1 = up   1 = down
		self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
 
		self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
		self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

	def gen_action(self,optfromNN):
		if optfromNN[0]==1: return MOVE_STAY
		elif optfromNN[1]==1: return MOVE_LEFT
		elif optfromNN[2]==1: return MOVE_RIGHT

	# action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
	# ai控制棒子左右移動；返回遊戲介面圖元數和對應的獎勵。(圖元->獎勵->強化棒子往獎勵高的方向移動)
	def step(self, action):
		if action == MOVE_LEFT:
			self.bar_pos_x = self.bar_pos_x - BALL_SPEED
			print("◀◀◀◀◀◀◀◀◀◀◀◀◀----------")
		elif action == MOVE_RIGHT:
			self.bar_pos_x = self.bar_pos_x + BALL_SPEED
			print("----------▷▷▷▷▷▷▷▷▷▷▷▷▷")
		elif action == MOVE_STAY:
			pass
		else:
			print("Not a action! ",action)
			pass
		if self.bar_pos_x < 0:
			self.bar_pos_x = 0
		if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
			self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
			
		self.screen.fill(BLACK)
		# draw bar
		self.bar_pos.left = self.bar_pos_x
		pygame.draw.rect(self.screen, WHITE, self.bar_pos)
 
		# draw ball
		self.ball_pos.left += self.ball_dir_x * BALL_SPEED
		self.ball_pos.bottom += self.ball_dir_y * BALL_SPEED
		if self.ball_dir_y==-1:
			pygame.draw.rect(self.screen, WHITE, self.ball_pos,0)
		else:
			pygame.draw.rect(self.screen, WHITE, self.ball_pos,3)
 
		#打到邊界反彈時，行進斜率 *=-1
		if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
			self.ball_dir_y = self.ball_dir_y * -1
		if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
			self.ball_dir_x = self.ball_dir_x * -1

		terminal = False
		reward = ALIVE_REWARD  # 存活獎勵       
		if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
			reward = WIN_REWARD    # 擊中獎勵
		elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
			reward = LOSE_REWARD   # 沒擊中懲罰
			terminal = True #落地死 
 
		# 獲得遊戲畫面的影像
		screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
		pygame.display.update()
		# 返回遊戲畫面和對應的賞罰
		return screen_image,reward, terminal

##################################################################################################################
##################################################################################################################

# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	#plt.imshow(observation, cmap ='gray'); plt.show();
	return np.reshape(observation,(80,80,1))

	
def playGame():
	# Step 0: Define reort
	win = 0
	lose = 0
	points = 0
	# Step 1: init BrainDQN
	actions = 3
	brain = BrainDQN(actions)
	# Step 2: init Game
	bg = Game()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = bg.gen_action([1,0,0])  # do nothing
	observation0, reward0, terminal = bg.step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	# Step 3.2: run the game
	while True:
		pygame.event.get()  #讓遊戲畫面能夠更新
		action = bg.gen_action(brain.getAction())
		Observation,reward,terminal = bg.step(action)
		nextObservation = preprocess(Observation)
		brain.setPerception(nextObservation,action,reward,terminal)
		
		########################  統計輸出報表用  ########################
		if reward==WIN_REWARD: win+=1 
		elif reward==LOSE_REWARD: lose+=1
		points+=reward
		if terminal==True: 
			points=0
			print("Game over~~")
		print("hit rate:" ,round(win/(win+lose+1)*100,2),"% ,win_points:",round(points,2)," ,cnt:",brain.timeStep)
		if (win+lose)>100:
			learn_rate.append(round(win/(win+lose+1)*100,2))
			plt.plot(learn_rate);plt.show();
			win=0
			lose=0
		########################  統計輸出報表用  ########################

		
learn_rate=[]
def main():
	playGame()

if __name__ == '__main__':
	main()
	
	
	
