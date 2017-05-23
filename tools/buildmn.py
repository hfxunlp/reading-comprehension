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

def handle(srctf, srcrf, rsf):
	ans = ldans(srcrf)
	cache = []
	curid = 0
	curans = ans["<qid_"+str(curid)+">"]
	ansid = 0
	curlid = 1
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						fwrt.write("\n".join(cache).encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						if ansid == 0:
							ansid = str(curlid)
						else:
							ansid = str(ansid)
						fwrt.write((str(curlid)+" "+"\t".join(("XXXXX? ", curans, ansid))).encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						cache = []
						curid += 1
						curans = ans.get("<qid_"+str(curid)+">", "XXXXX")
						ansid = 0
						curlid = 1
					else:
						tmp = tmp[tmp.find("|||")+4:]
						if tmp.find("?") != -1:
							tmp = tmp.replace("?", u"？")
						cache.append(" ".join((str(curlid), tmp,)))
						if curans in set(tmp.split(" ")):
							ansid = curlid
						curlid += 1

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
