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
		#string = re.sub(r"[\s：、。，,.;：；_:()-?!@#^&……（）\]\[]", " ", string)
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
				if self.add_feature == 1:
					content.extend(self.getFrom(self.generateindex(i),self.generateindex(j)))
				else:
					pass
				self.emaillist.append(content)
				j += 1
		for i in range(120):
			string = self.getword("215",self.generateindex(i))
			content = string.split()
			content = [con for con in content if len(con) >= 2]
			if self.add_feature == 1:
				content.extend(self.getFrom(self.generateindex(215),self.generateindex(i)))
			else:
				pass
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
	def __init__(self,Dataset,randomseed,percent):
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
		self.trainpercent = percent
		self.spamsum = 0
		self.hamsum = 0
		self.randomseed = randomseed

	def shufflelist(self):
		np.random.seed(self.randomseed)
		np.random.shuffle(self.indexlist)

	def cutfive(self,times):
		point_1 = self.total
		point = point_1//5
		self.testindex = self.indexlist[point*(times-1):point*times-1]
		self.trainindex = np.append(self.indexlist[:point*(times-1)],self.indexlist[point*times:])

	def cut(self,percent):
		self.percent = percent
		point = int(self.total * self.percent)
		self.trainindex = self.indexlist[:point]
		self.testindex = range(self.total)

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
		for i in self.spamdict.values():
			self.spamsum += i
		for i in self.hamdict.values():
			self.hamsum += i

	def testone(self):
		spam_correct_times = 0
		ham_correct_times = 0
		spamrealtotal = 0
		spamthinktotal = 0
		if self.trainpercent > 0:
			testnum = self.total
		else:
			testnum = self.total//5
		for i in self.testindex:
			logspam = 0.0
			logham = 0.0
			templist = list(set(self.emaillist[i]))
			for word in templist:
				if word in self.spamdict.keys():
					logspam += math.log((self.spamdict[word]+1) / (self.spamdict[word] + self.spamtimes+2))
				else:
					logspam += math.log(1 / (self.spamtimes+1))
				if word in self.hamdict.keys():
					logham += math.log((self.hamdict[word]+1) / (self.hamdict[word] + self.hamtimes+2))
				else:
					logham += math.log(1 / (self.spamtimes+2))
			judge = True
			logspam += math.log(self.spamtimes / (self.spamtimes + self.hamtimes))
			logham += math.log(self.hamtimes / (self.hamtimes + self.spamtimes))
			if logspam > logham:
				judge = True
			else:
				judge = False
			if self.labels[i] == '0':
				spamrealtotal += 1
			if judge:
				spamthinktotal += 1
			if judge == True and self.labels[i] == '0':
				spam_correct_times += 1
			if judge == False and self.labels[i] == '1':
				ham_correct_times += 1
		self.hamdict.clear()
		self.spamdict.clear()
		correct_times = spam_correct_times+ham_correct_times
		accuracy = correct_times / testnum
		precision = spam_correct_times / spamthinktotal
		recall = spam_correct_times / spamrealtotal
		F1_score = 2 * precision * recall/(precision+recall)
		self.spamtimes = 0
		self.hamtimes = 0
		self.hamsum = 0
		self.spamsum = 0
		return accuracy, precision, recall, F1_score

	def test(self):
		accuracy = 0
		precision = 0
		recall = 0
		F1 = 0
		print("Training the model...")
		self.shufflelist()
		for i in tqdm.tqdm(range(5)):
			self.cutfive((i+1))
			self.train()
			oneaccuracy,oneprecision,onerecall,oneF1 = self.testone()
			accuracy += oneaccuracy
			precision += oneprecision
			recall += onerecall
			F1 += oneF1
		print("accuracy: ",accuracy/5)
		print("precision: ",precision/5)
		print("recall: ",recall/5)
		print("F1: ",F1/5)

	def testsize(self):
		accuracy = 0
		maxacc = 0
		minacc = 1.2
		for i in tqdm.tqdm(range(5)):
			self.shufflelist()
			self.cut(self.trainpercent)
			self.randomseed += 5
			self.train()
			oneaccuracy,_,_,_ = self.testone()
			accuracy += oneaccuracy 
			if oneaccuracy > maxacc:
				maxacc = oneaccuracy
			if oneaccuracy < minacc:
				minacc = oneaccuracy
		accuracy /= 5
		print("max:",maxacc)
		print("min:",minacc)
		print("average:",accuracy)

def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--filename",type=str,default="trec06c-utf8")
	parser.add_argument("-p","--percent",type=float,default=0)
	parser.add_argument("-r","--randomseed",type=int,default=2333)
	parser.add_argument("-d","--add_feature",type=int,default=1)
	args = parser.parse_args()

	data = Dataset(args.filename,args.add_feature)
	trainer = Trainer(data,args.randomseed,args.percent)
	if args.percent>0:
		trainer.testsize()
	else:
		trainer.test()

main()	