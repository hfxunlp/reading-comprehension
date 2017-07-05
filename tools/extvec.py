#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from random import random

def ld(fmap, minkeep = 5):
	rs = {"<unk>":"1"}
	with open(fmap) as frd:
		curid = 2
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				tmp = tmp.split(" ")
				c = int(tmp[0])
				if c >= minkeep:
					for wd in tmp[1:]:
						rs[wd] = str(curid)
						curid += 1
				else:
					break
	return rs

def handle(fmap, vecf, rsf, vsize):
	rs={}
	wd = ld(fmap)
	with open(vecf) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				ind = tmp.find(" ")
				w = tmp[:ind]
				if w and w in wd:
					rs[wd[w]]=tmp[ind+1:]
	unkvec = rs.get(1, " ".join([str(random()) for i in xrange(vsize)]))
	with open(rsf, "w") as f:
		for i in xrange(1, len(wd)+1):
			f.write(rs.get(i, unkvec))
			f.write("\n")

if __name__=="__main__":
	handle(sys.argv[1].decode("utf-8"),sys.argv[2].decode("utf-8"),sys.argv[3].decode("utf-8"),int(sys.argv[4].decode("utf-8")))
