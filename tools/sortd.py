#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def handle(srctf, rsf):
	rsd = {}
	cache = []
	curlen = 0
	with open(srctf) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if tmp.startswith("<qid_"):
					cache.append(tmp)
					curlen += len(tmp.split(" "))
					if curlen in rsd:
						rsd[curlen].append("\n".join(cache))
					else:
						rsd[curlen] = ["\n".join(cache)]
					cache = []
					curlen = 0
				else:
					cache.append(tmp)
					curlen += len(tmp.split(" "))
	l = rsd.keys()
	l.sort()
	with open(rsf, "w") as f:
		for lu in l:
			f.write("\n".join(rsd[lu]).encode("utf-8"))
			f.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
