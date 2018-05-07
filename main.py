from random import random, choice, randint
from math import sin, cos, pi
from copy import deepcopy
from scipy.spatial import KDTree
from PIL import Image, ImageDraw
import numpy as np
import imageio

IMG_SCALE = 10
MAX_IO_LEN = 10
TAPE_SIZE = 128
NUM_AGENTS = 20
WORLD_SIZE = 40
WORLD_DIM = 2

class Agent:
	def __init__(self, rot=None, pos=None, code=None):
		self.energy = 100
		if rot is None:
			rot = rs(2*pi)
		self.rot = rot
		if pos is None:
			pos = rp()
		self.pos = pos
		if code is None:
			code = codegen()
		self.code = code
		self.tape = [0 for i in range(TAPE_SIZE)]
		self.ip = 0
		self.pt = 0
		self.inp = []
		self.out = []
		self.near = 0
		self.reproduce = False
		self.buildmap()
	
	def buildmap(self):
		self.map = {}
		stack = []
		for pos, com in enumerate(self.code):
			if com == "[":
				stack.append(pos)
			elif com == "]":
				if len(stack) == 0:
					start = 0
				else:
					start = stack.pop()
				self.map[start] = pos
				self.map[pos] = start
		
		for el in stack:
			self.map[el] = len(self.code)-1
	
	def distance(self, agent):
		return sum((self.pos[i]-agent.pos[i])**2 for i in range(WORLD_DIM))
	
	def push(self, mem):
		self.inp.append(mem)
		self.inp = self.inp[:MAX_IO_LEN]
	
	def run(self, steps):
		for i in range(steps):
			#print(self.code, self.ip, self.map)
			codelen = len(self.code)
			ipo = self.ip%codelen
			com = self.code[ipo]
			tapelen = len(self.tape)
			self.reproduce = False
			if com == ">":
				self.pt += 1
			elif com == "<":
				self.pt -= 1
			elif com == "+":
				self.tape[self.pt%tapelen] += 1
			elif com == "-":
				self.tape[self.pt%tapelen] -= 1
			elif com == "[":
				if self.tape[self.pt%tapelen] == 0:
					self.ip = self.map[ipo]
			elif com == "]":
				if self.tape[self.pt%tapelen] != 0:
					self.ip = self.map[ipo]	
			elif com == ".":
				self.out.append(self.tape[self.pt%tapelen])
			elif com == ",":
				if len(self.inp) > 0:
					self.tape[self.pt%tapelen] = self.inp.pop(0)
			elif com == "l":
				self.rot -= 0.01
			elif com == "r":
				self.rot += 0.01
			elif com == "f":
				self.pos[0] += sin(self.rot) * 0.1
				self.pos[1] += cos(self.rot) * 0.1
			elif com == "x":
				if self.energy > 300:
					self.reproduce = True
			elif com == "n":
				self.tape[self.pt%tapelen] = self.near
			"""
			elif com == "c":
				if self.energy > 200:
					self.energy -= 200
					child = deepcopy(self)
					child.rot = rs(2*pi)
					child.energy = 100
					agents.append(child)
			"""
			self.ip += 1

# Random scaled number
def rs(scale):
	return random()*scale

# Random vector with specified dimension
def rv(scale, dimension):
	return [rs(scale) for d in range(dimension)]

# Random position
def rp():
	return rv(WORLD_SIZE, WORLD_DIM)

# Scale vector
def sv(scale, v):
	return [e*scale for e in v]

# Modulo vector
def mv(mod, v):
	return [e%mod for e in v]

# Average vector
def av(v1, v2):
	return [(v1[i]+v2[i])/2 for i in range(len(v1))]

# Rectangle coordinates
def rect(pos, scale):
	return [pos[0]-scale,pos[1]-scale,pos[0]+scale,pos[1]+scale]

# Generate random code sequence
instr = "><+-.,[]lrfcxn"
def codegen():
	return "".join(choice(instr) for i in range(randint(1,100)))

# Converts PIL image to numpy array
def PIL2array(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 3)

# Chooses character at random
def mutate_random(s1, s2):
	result = ""
	minl = min(len(s1),len(s2))
	maxl = max(len(s1),len(s2))
	for i in range(randint(minl,maxl)):
		s = []
		if i<len(s1):
			s.append(s1[i])
		if i<len(s2):
			s.append(s2[i])
		result += choice(s)
	return result

# Cuts two strings to the same length
def string_samelen(s1, s2):
	if len(s1) < len(s2):
		s2 = s2[:len(s1)]
	if len(s1) > len(s2):
		s1 = s1[:len(s2)]
	return s1, s2

# Splits slices
def mutate_split(s1, s2):
	result = ""
	
	split = randint(0,min(len(s1), len(s2)))
	if randint(0,1) == 0:
		return s1[:split]+s2[split:]
	else:
		return s2[:split]+s1[split:]

# Mixes/mutates two strings
def mutate(s1, s2):
	return mutate_split(s1, s2)

images = []

agents = [Agent() for i in range(NUM_AGENTS)]
food = [rp() for i in range(1000)]


w = h = WORLD_SIZE*IMG_SCALE

def step(count):
	global agents

	im = Image.new("RGB", (w,h), color="white")
	draw = ImageDraw.Draw(im)

	for f in food:
		draw.rectangle(sv(IMG_SCALE, rect(f, 0.05)), fill="green")

	tree = KDTree(food)
	agtree = KDTree([agent.pos for agent in agents])
	
	if i%10 == 0:
		print(len(agents))
		food.append(rp())

	remove = []
	for agentindex, agent in enumerate(agents):
		agent.run(24)
		agent.pos = mv(WORLD_SIZE, agent.pos)
		near = tree.query_ball_point(agent.pos, 0.5)
		for index in near:
			try:
				food.pop(index)
			except IndexError:
				pass
			agent.energy += 20
		agent.energy -= 1
		if agent.energy < 0:
			remove.append(agentindex)
			continue
		rwh = 0.1
		draw.rectangle(sv(IMG_SCALE, rect(agent.pos, rwh)), fill="black")

		if agent.reproduce:
			near_agents = agtree.query_ball_point(agent.pos, 5)
			if len(near_agents) > 0:
				#print("reproduced!")
				other_agent = agents[near_agents[0]]
				new_pos = av(agent.pos, other_agent.pos)
				new_code = mutate(agent.code, other_agent.code)
				new_agent = Agent(pos=new_pos, code=new_code)
				agents.append(new_agent)

		near_agents = agtree.query_ball_point(agent.pos, 10)
		agent.near = len(near_agents)
		for b in near_agents:
			if b != agentindex:
				other_agent = agents[b]
				for call in agent.out:
					other_agent.push(call)
		agent.out = []
	#im.save("anim/%i.jpg" % i)
	agents = [agent for agentindex, agent in enumerate(agents) if agentindex not in remove]
	if count % 5 == 0:
		images.append(PIL2array(im))

try:
	for i in range(1000000):
		step(i)
except KeyboardInterrupt:
	pass

imageio.mimsave("anim.gif", images)
