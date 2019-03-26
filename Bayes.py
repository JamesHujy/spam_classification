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
	def __init__(self, filename, add_feature):
		print("Loading the emails...")
		self.filename = filename
		self.emaillist = []
		self.labels = []
		self.spamtimes = 0
		self.hamtimes = 0
		self.Fromlist = []
		self.timelist = []
		self.add_feature = add_feature

	def generateindex(self, index):
		if index < 10:
			return "00"+str(index)
		elif index <100:
			return "0" + str(index)
		else:
			return str(index)

	def getword(self, index1, index2):
		path = "./"+self.filename+'/data_cut/'+index1+"/"+index2
		f = codecs.open(path,encoding='utf-8')
		string = f.read()
		if self.add_feature:
			string = re.sub(r"[\s：、。，,.;：；_:()-?!@#^&……（）\]\[]", " ", string)
		else:
			string = re.sub(r"[\s：、。，,.;：；_★:()-?!@#^&$¥……（）◆\]\[]", " ", string)
		string = re.sub(r"[a-zA-Z]"," ",string)
		return string

	def getFrom(self, index1, index2):
		path = "./"+self.filename+'/data_cut/'+index1+"/"+index2
		f = codecs.open(path,encoding='utf-8')
		string = f.readlines()
		i = 0
		while string[i][:4] != "From":
			i += 1
			if i >= len(string):
				return []
		Fromstring = string[i]
		Fromstring = re.findall("@\w*",Fromstring)
		return Fromstring

	def readlist(self):
		for i in tqdm.tqdm(range(215)):
			j = 0
			while j < 300:
				string = self.getword(self.generateindex(i),self.generateindex(j))
				content = string.split()
				content = [con for con in content if len(con) >= 2]
				if self.add_feature:
					content.extend(self.getFrom(self.generateindex(i),self.generateindex(j)))
				content = list(set(content))
				self.emaillist.append(content)
				j += 1
		for i in range(120):
			string = self.getword("215",self.generateindex(i))
			content = string.split()
			content = [con for con in content if len(con) >= 2]
			if self.add_feature:
					content.extend(self.getFrom(self.generateindex(215),self.generateindex(i)))
			content = list(set(content))
			self.emaillist.append(content)
		return self.emaillist

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
		spam_correct_times = 0
		ham_correct_times = 0
		spamtotal = 0
		hamtotal = 0
		testnum = self.total * self.trainpercent/ 500
		for i in self.testindex:
			logspam = 0.0
			logham = 0.0
			for word in self.emaillist[i]:
				if word in self.spamdict.keys():
					logspam += math.log(self.spamdict[word] / self.spamtimes)
				else:
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
			if self.labels[i] == '0':
				spamtotal += 1
			else:
				hamtotal += 1
			if judge == True and self.labels[i] == '0':
				spam_correct_times += 1
			if judge == False and self.labels[i] == '1':
				ham_correct_times += 1
		self.hamdict.clear()
		self.spamdict.clear()
		correct_times = spam_correct_times+ham_correct_times
		accuracy = correct_times / testnum
		precision = ham_correct_times / hamtotal
		recall = spam_correct_times / spamtotal
		F1_score = 2 * precision * recall/(precision+recall)
		self.spamtimes = 0
		self.hamtimes = 0
		return accuracy, precision, recall, F1_score

	def test(self):
		accuracy = 0
		precision = 0
		recall = 0
		F1 = 0
		accuracylist = []
		print("Training the model...")
		for i in tqdm.tqdm(range(5)):
			self.cut((i+1))
			self.train()
			oneaccuracy,oneprecision,onerecall,oneF1 = self.testone()
			accuracy += oneaccuracy
			precision += oneprecision
			recall += onerecall
			F1 += oneF1
			accuracylist.append(oneaccuracy)
		print("accuracy: ",accuracy/5)
		print("precision: ",precision/5)
		print("recall: ",recall/5)
		print("F1: ",F1/5)

def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--filename",type=str,default="trec06c-utf8")
	parser.add_argument("-p","--percent",type=int,default=100)
	parser.add_argument("-r","--randomseed",type=int,default=233)
	parser.add_argument("-d","--add_feature",type=bool,default=False)
	args = parser.parse_args()

	data = Dataset(args.filename,args.add_feature)
	trainer = Trainer(data)
	trainer.decidetrain(args.randomseed,args.percent)
	trainer.test()

main()	