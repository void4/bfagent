from random import random, choice, randint
from math import sin, cos, pi
from copy import deepcopy
from scipy.spatial import KDTree
import numpy as numpy
import imageio

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
			elif com == "c":
				if self.energy > 200:
					self.energy -= 200
					child = deepcopy(self)
					child.rot = rs(2*pi)
					child.energy = 100
					agents.append(child)
			
			self.ip += 1

def rs(scale):
	return random()*scale

def rv(scale, dimension):
	return [rs(scale) for d in range(dimension)]

instr = "><+-.,[]lrfc"
def codegen():
	return "".join(choice(instr) for i in range(randint(1,100)))

# Random position
def rp():
	return rv(WORLD_SIZE, WORLD_DIM)

agents = [Agent() for i in range(NUM_AGENTS)]
food = [rp() for i in range(1000)]
from PIL import Image, ImageDraw
IMG_SCALE = 10
w = h = WORLD_SIZE*IMG_SCALE

# Scale vector
def sv(scale, v):
	return [e*scale for e in v]

# Modulo vectro
def mv(mod, v):
	return [e%mod for e in v]

def rect(pos, scale):
	return [pos[0]-scale,pos[1]-scale,pos[0]+scale,pos[1]+scale]



def PIL2array(img):
    return numpy.array(img.getdata(),
                    numpy.uint8).reshape(img.size[1], img.size[0], 3)

images = []

def step():

	im = Image.new("RGB", (w,h), color="white")
	draw = ImageDraw.Draw(im)

	for f in food:
		draw.rectangle(sv(IMG_SCALE, rect(f, 0.05)), fill="green")

	tree = KDTree(food)
	
	if i%10 == 0:
		print(len(agents))
		food.append(rp())

	for agent in agents:
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
			agents.remove(agent)
			continue
		rwh = 0.1
		draw.rectangle(sv(IMG_SCALE, rect(agent.pos, rwh)), fill="black")
		for b in agents:
			if b != agent:
				if b.distance(agent) < 10:
					for call in agent.out:
						b.push(call)
	#im.save("anim/%i.jpg" % i)
	images.append(PIL2array(im))

try:
	for i in range(1000000):
		step()
except KeyboardInterrupt:
	pass

imageio.mimsave("anim.gif", images)
