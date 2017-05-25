#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def getans_mwd(pas, scl):
	tmp = scl.strip()
	tmp = tmp.decode("utf-8")
	s = [float(tmpu) for tmpu in tmp.split()]
	rsd = {}
	if s:
		for sent in pas:
			cnt = False
			for wd in sent:
				tmp = wd.strip()
				if tmp:
					if not tmp in rsd and s:
						rsd[tmp] = s[0]
						del s[0]
						if not s:
							cnt=True
							break
			if cnt:
				break
		rsw = rsd.keys()[0]
		mscore = rsd[rsw]
		for w, score in rsd.iteritems():
			if score > mscore:
				mscore = score
				rsw = w
		return rsw, mscore
	else:
		return "err", -99999.9

def getans(pas, scl):
	w, s = getans_mwd(pas, scl)
	#w, s = getans_cwd(pas, scl)
	return w

def handle(srcif, srctf, rsf):
	rs = []
	cache = []
	curid = 0
	with open(srcif) as frdi:
		with open(srctf) as frdt:
			for line in frdi:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						rs.append(" ".join(["<qid_"+str(curid)+">", "|||", getans(cache, frdt.readline())]))
						cache = []
						curid += 1
					else:
						tmp = tmp[tmp.find("|||")+4:]
						cache.append(tmp.split())
	if cache:
		rs.append(" ".join(["<qid_"+str(curid)+">", "|||", getans(cache, frdt.readline())]))
		cache = []
	rs = "\n".join(rs)
	with open(rsf, "w") as fwrt:
		fwrt.write(rs.encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
