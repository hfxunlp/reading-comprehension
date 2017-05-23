#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def ldans(fname):
	rs = {}
	with open(fname) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				key, v = tmp.decode("utf-8").split(" ||| ")
				rs[k] = v
	return rs

def handle(srctf, srcrf, rsf):
	ans = ldans(srcrf)
	cache = []
	curid = 0
	curans = ans["<qid_"+str(curid)+">"]
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						fwrt.write("\n".join(cache).encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						curid += 1
						curans = ans["<qid_"+str(curid)+">"]
					else:
						tmp = tmp[tmp.find("|||")+4:].replace("XXXXX", curans)
						cache.append(tmp)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
