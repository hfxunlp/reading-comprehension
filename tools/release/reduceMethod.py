#encoding: utf-8

import sys

def isForward(strin):
	tmp=strin.strip()
	if strin.startswith("function ") and (tmp.find(":updateOutput(")!=-1 or tmp.find(":forward(")!=-1) and tmp.endswith(")"):
		return True
	else:
		return False

def isGlobalFunc(strin):
	if (strin.startswith("function ") or strin.startswith("return function ")) and strin.find(":")==-1 and strin.strip().endswith(")"):
		return True
	else:
		return False

def isKeepFunc(strin):
	rs=True
	if strin.startswith("function ") and strin.strip.endswith(")"):
		fbd=[":__init(", ":backward(", ":updateGradInput(", ":accGradParameters(", ":accUpdateGradParameters(", ":defaultAccUpdateGradParameters(", ":sharedAccUpdateGradParameters(", ":zeroGradParameters(", ":updateParameters(", ":training(", ":evaluate(", ":clearState("]
		for fu in fbd:
			if strin.find(fu)!=-1:
				rs=False
				break
	else:
		rs=False
	return rs

def handle(srcf, rsf):
	cf=False
	nd=True
	with open(srcf) as frd:
		with open(rsf, "w") as fwrt:
			for line in frd:
				if line.find("torch.class")!=-1:
					cf=True
					fwrt.write(line)
					nd=False
				if cf:
					if line.startswith("local function") or isKeepFunc(strin) or isForward(line) or isGlobalFunc(line):
						nd=True
					elif line.startswith("end"):
						fwrt.write(line)
						nd=False
					if nd or line.strip().startswith("require \""):
						fwrt.write(line)
				else:
					fwrt.write(line)
