from random import random, choice, randint
from math import sin, cos, pi
from copy import deepcopy
from scipy.spatial import KDTree
from PIL import Image, ImageDraw
import numpy as np
import imageio
import bisect



IMG_SCALE = 10
MAX_IO_LEN = 10
TAPE_SIZE = 128
NUM_AGENTS = 30
WORLD_SIZE = 50
WORLD_DIM = 2

class Agent:
	def __init__(self, rot=None, pos=None, code=None):
		self.energy = 600
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
		self.age = 0
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
		codelen = len(self.code)
		tapelen = len(self.tape)

		self.age += 1

		self.energy -= 10
		self.action = None
		
		self.attack = False
		self.reproduce = False

		for i in range(steps):
			#print(self.code, self.ip, self.map)

			ipo = self.ip%codelen
			com = self.code[ipo]


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
				self.action = [".", self.tape[self.pt%tapelen]]
			elif com == ",":
				if len(self.inp) > 0:
					self.tape[self.pt%tapelen] = self.inp.pop(0)
			elif com == "l":
				self.action = "l"
			elif com == "r":
				self.action = "r"
			elif com == "f":
				self.action = "f"
			elif com == "x":
				self.action = "x"
			elif com == "n":
				self.tape[self.pt%tapelen] = self.near
			elif com == "a":
				self.action = "a"
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
		
		if self.action is None:
			pass
		elif self.action == "f":
			self.pos[0] += sin(self.rot) * 0.1
			self.pos[1] += cos(self.rot) * 0.1
			self.energy -= 3
		elif self.action == "l":
			self.rot -= 0.01
			self.energy -= 2
		elif self.action == "r":
			self.rot += 0.01
			self.energy -= 2
		elif self.action == "x":
			if self.energy > 800:		
				self.reproduce = True
		elif self.action == "a":
			self.attack = True
			#self.energy -= 4
		elif self.action[0] == ".":
			self.out.append(self.action[1])
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

# Implements weighted choice selection
def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def weighted_choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]

instr = "><+-.,[]lrfxn"#c
weights = [100,100,100,100,4,4,10,10,7,7,10,7,10]
combined = list(zip(instr, weights))
MAXCODELEN = 128
def gen_weighted():
	return "".join(weighted_choice(instr, weights) for i in range(randint(1,MAXCODELEN)))

conflicts = ["+-", "-+", "<>", "><", "rl", "lr", "[]"]

def gen_alt():
	code = choice(instr)
	for i in range(MAXCODELEN):
		allowed = list(combined)
		for pair in conflicts:
			if code[-1] == pair[0]:
				allowed.remove(combined[instr.index(pair[1])])
		c,w = list(zip(*allowed))
		#print(code[-1], c)
		code += weighted_choice(c,w)
	#print(code)
	return code


# Optimizes bfagent code
def optimize(code):
	old = code
	for i in range(len(code)//2):
		opt = ""
		i = 0
		while i < len(old)-1:
			if code[i]+old[i+1] in ["+-", "-+", "><", "<>"]:
				print(code[i]+old[i+1])
				i += 2
			else:
				opt += old[i]
				i += 1
	
		old = opt
	print(len(code)-len(old))
	return old

# Generate random code sequence

def codegen():
	"""
	while True:
		code = optimize(gen_weighted())
		if code:
			return code
	"""
	return gen_alt()

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
food = [rp() for i in range(WORLD_SIZE**2*5)]


w = h = WORLD_SIZE*IMG_SCALE

def step(count):
	global agents

	im = Image.new("RGB", (w,h), color="white")
	draw = ImageDraw.Draw(im)

	for f in food:
		draw.rectangle(sv(IMG_SCALE, rect(f, 0.05)), fill="green")

	tree = KDTree(food)
	agtree = KDTree([agent.pos for agent in agents])
	#print(i)
	if i%10 == 0:
		energies = [agent.energy for agent in agents]
		ages = [agent.age for agent in agents]
		print("NumAgt:%i\tAvgAge:\t%i\tAvgEng:%i\tStdDev%i" % (len(agents), np.mean(ages), np.mean(energies), np.std(energies)))
		food.append(rp())
	
	if len(agents) < NUM_AGENTS:
		agents.append(Agent())

	remove = []
	for agentindex, agent in enumerate(agents):
		agent.run(1024)
		agent.pos = mv(WORLD_SIZE, agent.pos)
		near = tree.query_ball_point(agent.pos, 0.2)
		#print(len(near))
		for index in near:
			try:
				food.pop(index)
			except IndexError:
				pass
			agent.energy += 80
		agent.energy -= 1
		if agent.energy < 0 or agent.age**1.2 * 10 > agent.energy:
			remove.append(agentindex)
			continue
		rwh = 0.1
		draw.rectangle(sv(IMG_SCALE, rect(agent.pos, rwh)), fill="black")

		if agent.attack:
			near_agents = agtree.query_ball_point(agent.pos, 4)
			try:
				near_agents.remove(agentindex)
			except ValueError:
				pass
			if len(near_agents) > 0:
				agent.energy -= 12
				other_agent = agents[near_agents[0]]
				other_agent.energy -= 8

		if agent.reproduce:
			near_agents = agtree.query_ball_point(agent.pos, 1.5)
			try:
				near_agents.remove(agentindex)
			except ValueError:
				pass
			if len(near_agents) > 0:
				print("reproduced!", len(near_agents))
				agent.energy -= 700
				other_agent = agents[near_agents[0]]
				new_pos = av(agent.pos, other_agent.pos)
				new_code = mutate(agent.code, other_agent.code)
				new_agent = Agent(pos=new_pos, code=new_code)
				agents.append(new_agent)

		near_agents = agtree.query_ball_point(agent.pos, 3)
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
