# coding=utf-8
import re
import numpy as np 
import random
import math
import tqdm
import codecs

def generateindex(index):
	if index < 10:
		return "00"+str(index)
	elif index <100:
		return "0" + str(index)
	else:
		return str(index)

def getword(index1,index2):
	path = "trec06c-utf8/data_cut/"+index1+"/"+index2
	f = codecs.open(path,encoding='utf-8')
	str = f.read()
	str = re.sub(r"[\s：、。，,.;：；_★:()-?!@#^&$¥……（）◆\]\[]", " ", str)
	str = re.sub(r"[a-zA-Z]"," ",str)
	return str

def get215word(index):
	path = "trec06c-utf8/data_cut/215/"+index
	f = codecs.open(path,encoding='utf-8')
	str = f.read()
	str = re.sub(r"[\s：、。，,.;：；_★:()-?!@#^&$¥……（）◆\]\[]", " ", str)
	str = re.sub(r"[a-zA-Z]"," ",str)
	return str
emaillist = [] 
labels = []
trainindex = []
testindex = []

def decidetrain(randomseed, trainpercent):
	global trainindex
	global testindex
	total = 215*300+119
	indexlist = list(range(total))
	random.shuffle(indexlist)
	point_1 = int(total*trainpercent)
	point = int(total/5*4)
	trainindex = indexlist[0:point_1-1]
	testindex = indexlist[point:-1]

def getwordlist():
	print("processing the words...")	
	global emaillist
	for i in tqdm.tqdm(range(215)):
		j = 0
		while j < 300:
			str = getword(generateindex(i),generateindex(j))
			content = str.split()
			content = [con for con in content if len(con) >= 2]
			emaillist.append(content)
			j += 1
			#if times % 1000 == 0:
				#print("processing...finsihed %",times/215/3)
	for i in range(120):
		str = get215word(generateindex(i))
		content = str.split()
		content = [con for con in content if len(con) >= 2]
		emaillist.append(content)

spamtimes = 0
hamtimes = 0
def getlabels():
	path = "trec06c-utf8/label/index"
	global labels
	global spamtimes
	global hamtimes
	f = open(path)
	lines = f.readlines()
	for i in lines:
		if i[0] == 's':
			labels.append('0')
			spamtimes += 1
		else:
			labels.append('1')
			hamtimes += 1
	return spamtimes,hamtimes
spamdict ={}
hamdict = {}

def trainer():
	global emaillist
	global labels
	global spamdict
	global hamdict
	for i in trainindex:
		for word in emaillist[i]:
			if labels[i] == '0':
				if word in spamdict.keys():
					spamdict[word] += 1
				else:
					spamdict[word] = 1
			else:
				if word in hamdict.keys():
					hamdict[word] += 1
				else:
					hamdict[word] = 1

def test():
	correct_times = 0
	total = 215*300+119
	testnum = total//5
	global emaillist
	for i in testindex:
		logspam = 0.0
		logham = 0.0
		for word in emaillist[i]:
			if word in spamdict.keys():
				logspam += math.log(spamdict[word]/spamtimes)
				#print(word,"'s spamfre:",spamdict[word])
			else:
				logspam += math.log(1/(spamtimes+2))
				#print(word,"'s spamfre:",0)
			if word in hamdict.keys():
				logham += math.log(hamdict[word]/hamtimes)
				#print(word,"'s hamfre:",hamdict[word])
			else:
				logham += math.log(1/(hamtimes+2))
				#print(word,"'s hamfre:",0)
		judge = True
		logspam += math.log(spamtimes/(spamtimes+hamtimes))
		logham += math.log(hamtimes/(hamtimes+spamtimes))
		if logspam > logham:
			judge = True
		else:
			judge = False
		if judge == True and labels[i] == '0':
			correct_times += 1
		if judge == False and labels[i] == '1':
			correct_times += 1
	print(correct_times/testnum)

decidetrain(908, 80)
getlabels()
getwordlist()
trainer()
test()





	