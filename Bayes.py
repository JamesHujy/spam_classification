# coding=utf-8
import re
import numpy as np
import random
import math
import tqdm
import codecs

class Dataset(object):
	"""
	build a reader class for readthe data
	"""
	def __init__(self, filename):
		print("processing the words...")
		self.filename = filename
		self.emaillist = []
		self.labels = []
		self.spamtimes = 0
		self.hamtimes = 0
		self.Fromlist = []
		self.timelist = []

	def generateindex(self, index):
		if index < 10:
			return "00"+str(index)
		elif index <100:
			return "0" + str(index)
		else:
			return str(index)

	def getword(self, index1, index2):
		path = self.filename+'/data_cut/'+index1+"/"+index2
		f = codecs.open(path,encoding='utf-8')
		string = f.read()
		string = re.sub(r"[\s：、。，,.;：；_★:()-?!@#^&$¥……（）◆\]\[]", " ", string)
		string = re.sub(r"[a-zA-Z]"," ",string)
		return string

	def gettime(self, index1, index2):
		path = self.filename+'/data_cut/'+index1+"/"+index2
		f = codecs.open(path,encoding='utf-8')
		string = f.readlines()
		timestring = string[2]
		timestring = re.findall("\d\d:\d\d:\d\d",timestring)
		return timestring

	def getFrom(self, index1, index2):
		
		path = self.filename+'/data_cut/'+index1+"/"+index2
		f = codecs.open(path,encoding='utf-8')
		string = f.readlines()
		i = 0
		while string[i][:4] != "From":
			i += 1
			if i >= len(string):
				return []
		Fromstring = string[i]
		Fromstring = re.findall("@\w*",Fromstring)
		finalFromstring = Fromstring[1:]
		print(finalFromstring)
		return finalFromstring

	def readlist(self):
		for i in tqdm.tqdm(range(215)):
			j = 0
			while j < 300:
				string = self.getword(self.generateindex(i),self.generateindex(j))
				content = string.split()
				content = [con for con in content if len(con) >= 2]
				content = list(set(content))
				self.emaillist.append(content)
				j += 1
		for i in range(120):
			string = self.getword("215",self.generateindex(i))
			content = string.split()
			content = [con for con in content if len(con) >= 2]
			content = list(set(content))
			self.emaillist.append(content)
		return self.emaillist

	def gettimelist(self):
		for i in tqdm.tqdm(range(215)):
			j = 0
			while j < 300:
				timestring = self.gettime(self.generateindex(i),self.generateindex(j))
				self.timelist.append(timestring)
				j += 1
		for i in range(120):
			timestring = self.gettime("215",self.generateindex(i))
			self.timelist.append(timestring)
		return self.timelist

	def getFromlist(self):
		print("Loading the Receiver email address")
		for i in tqdm.tqdm(range(215)):
			j = 0
			while j < 300:
				print(i," ",j)
				Fromstring = self.getFrom(self.generateindex(i),self.generateindex(j))
				self.Fromlist.append(Fromstring)
				j += 1
		for i in range(120):
			Fromtring = self.getFrom("215",self.generateindex(i))
			self.Fromlist.append(Fromstring)
		return self.Fromlist

	def getlabels(self):
		path = self.filename+"/label/index"
		f = open(path)
		lines = f.readlines()
		for i in lines:
			if i[0] == 's':
				self.labels.append('0')
				self.spamtimes += 1
			else:
				self.labels.append('1')
				self.hamtimes += 1
		return self.labels


class Trainer(object):
	def __init__(self,Dataset):
		self.trainindex = []
		self.testindex = []
		self.spamdict = {}
		self.hamdict = {}
		self.emaillist = Dataset.readlist()
		self.labels = Dataset.getlabels()
		self.timelist = Dataset.gettimelist()
		self.Fromlist = Dataset.getFromlist()
		#print(self.timelist)
		self.hamtimes = 0
		self.spamtimes = 0
		self.total = 215 * 300 + 119
		self.indexlist = np.arange(self.total)
		self.trainpercent = 1

	def decidetrain(self, randomseed, trainpercent):
		np.random.seed(randomseed)
		np.random.shuffle(self.indexlist)
		self.trainpercent = trainpercent

	def cut(self,times):
		point_1 = int(self.total * self.trainpercent/100)
		point = point_1//5
		self.testindex = self.indexlist[point*(times-1):point*times-1]
		self.trainindex = np.append(self.indexlist[:point*(times-1)],self.indexlist[point*times:])
		
	def train(self):
		for i in self.trainindex:
			if self.labels[i] == '0':
				self.spamtimes+=1
			else:
				self.hamtimes+=1
			for word in self.emaillist[i]:
				if self.labels[i] == '0':
					if word in self.spamdict.keys():
						self.spamdict[word] += 1
					else:
						self.spamdict[word] = 1
				else:
					if word in self.hamdict.keys():
						self.hamdict[word] += 1
					else:
						self.hamdict[word] = 1

	def testone(self):
		correct_times = 0
		testnum = self.total * self.trainpercent/ 500
		for i in self.testindex:
			logspam = 0.0
			logham = 0.0
			for word in self.emaillist[i]:
				if word in self.spamdict.keys():
					logspam += math.log(self.spamdict[word] / self.spamtimes)
					#logspam += math.log(self.spamdict[word] / self.spamtimes)
				else:
					#logspam += math.log(1 / (self.spamtimes + 2))
					logspam += math.log(1 / (self.spamtimes + 2))
				if word in self.hamdict.keys():
					logham += math.log(self.hamdict[word] / self.hamtimes)
				else:
					logham += math.log(1 / (self.hamtimes + 2))
			judge = True
			logspam += math.log(self.spamtimes / (self.spamtimes + self.hamtimes))
			logham += math.log(self.hamtimes / (self.hamtimes + self.spamtimes))
			if logspam > logham:
				judge = True
			else:
				judge = False
			if judge == True and self.labels[i] == '0':
				correct_times += 1
			elif judge == False and self.labels[i] == '1':
				correct_times += 1
			else:
				pass
				#print(i)
				#print(self.emaillist[i])
		self.spamtimes = 0
		self.hamtimes = 0
		self.hamdict.clear()
		self.spamdict.clear()
		accuracy = correct_times / testnum
		return accuracy

	def test(self):
		accuracy = 0
		accuracylist = []
		for i in range(5):
			self.cut((i+1))
			self.train()
			print(self.hamtimes," ",self.spamtimes)
			oneaccuracy = self.testone()
			accuracy += oneaccuracy
			accuracylist.append(oneaccuracy)
		print(accuracylist)	
		print(accuracy/5)

def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--filename",type=str,default="trec06c-utf8")
	parser.add_argument("-p","--percent",type=int,default=100)
	parser.add_argument("-r","--randomseed",type=int,default=100)
	args = parser.parse_args()

	data = Dataset(args.filename)
	trainer = Trainer(data)
	trainer.decidetrain(args.randomseed,args.percent)
	trainer.test()

main()





	