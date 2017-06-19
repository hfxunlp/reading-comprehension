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

def clearl(lin):
	rs = []
	for lu in lin:
		tt = lu.strip()
		if tt:
			rs.append(tt)
	return rs

def handle(srctf, srcrf, rsf):
	ans = ldans(srcrf)
	cache = []
	wds=set()
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						cache.append(tmp)
						curans = ans[tmp[:tmp.find(">") + 1]]
						if curans in wds:
							tmp="\n".join(cache)
							fwrt.write(tmp.encode("utf-8"))
							fwrt.write("\n".encode("utf-8"))
						cache = []
						wds.clear()
					else:
						cache.append(tmp)
						for wd in clearl(tmp[tmp.find("|||") + 4:].split(" ")):
							if not wd in wds:
								wds.add(wd)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
