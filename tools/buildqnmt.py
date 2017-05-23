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
				rs[key] = v
	return rs

def handle(srctf, srcrf, rsf1, rsf2):
	ans = ldans(srcrf)
	cache = []
	curid = 0
	curans = ans["<qid_"+str(curid)+">"]
	with open(rsf1, "w") as fwrts:
		with open(rsf2, "w") as fwrtt:
			with open(srctf) as frd:
				for line in frd:
					tmp = line.strip()
					if tmp:
						tmp = tmp.decode("utf-8")
						if tmp.startswith("<qid_"):
							cache.insert(0, tmp[tmp.find("|||")+4:])
							cache.append("")
							fwrts.write(" ".join(cache).encode("utf-8"))
							fwrtt.write(curans.encode("utf-8"))
							fwrtt.write("\n".encode("utf-8"))
							cache = []
							curid += 1
							curans = ans.get("<qid_"+str(curid)+">", "XXXXX")
						else:
							tmp = tmp[tmp.find("|||")+4:]
							cache.append(tmp)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"), sys.argv[4].decode("gbk"))
