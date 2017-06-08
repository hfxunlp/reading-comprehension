#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from pynlpir import nlpir

def segline(strin):
	try:
		rs=nlpir.ParagraphProcess(strin.encode("utf-8","ignore"), 1)
	except:
		rs=""
	return rs.decode("utf-8","ignore")

def ldmap(fmap, minkeep = 5):
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
						rs[wd] = curid
						curid += 1
				else:
					break
	return rs

def getans_mwd(pas, scl, mapd):
	tmp = scl.strip()
	tmp = tmp.decode("utf-8")
	s = [float(tmpu) for tmpu in tmp.split()]
	src = set()
	cache = set()
	for sent in pas:
		src |= set(sent)
		tmp = "".join(sent)
		tmp = tmp.replace(" ","")
		tmp = segline(tmp).split(" ")
		for tmpu in tmp:
			ind = tmpu.rfind("/")
			wd, tag = tmpu[:ind], tmpu[ind+1:]
			if tag.startswith("n") and not wd in cache:
				cache.add(wd)
	alw = set()
	for wd in cache:
		if wd in src:
			alw.add(wd)
	rsd = {}
	allwd = {}
	for sent in pas:
		cnt = False
		for wd in sent:
			tmp = wd.strip()
			if tmp:
				wid = mapd.get(tmp, 1)
				if not wid in allwd:
					if s:
						allwd[wid] = s[0]
						del s[0]
						if not s:
							cnt=True
							break
					else:
						cnt=True
						break
				if wid in allwd and tmp not in rsd and tmp!="XXXXX":
					rsd[tmp] = allwd[wid]
		if cnt:
			break
	rsw = rsd.keys()[0]
	mscore = rsd[rsw]
	lim = False
	for wd in alw:
		if wd in rsd:
			rsw = wd
			mscore = rsd[rsw]
			lim = True
			break
	for w, score in rsd.iteritems():
		if not lim or w in alw:
			if score > mscore:
				mscore = score
				rsw = w
	return rsw, mscore

def getans(pas, scl, mapd):
	w, s = getans_mwd(pas, scl, mapd)
	#w, s = getans_cwd(pas, scl)
	return w

def handle(mapf, srcif, srctf, rsf, minkeep = 5):
	rs = []
	cache = []
	curid = 0
	mapd = ldmap(mapf, minkeep)
	with open(srcif) as frdi:
		with open(srctf) as frdt:
			for line in frdi:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						rs.append(" ".join(["<qid_"+str(curid)+">", "|||", getans(cache, frdt.readline(), mapd)]))
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
	nlpir.Init(nlpir.PACKAGE_DIR,nlpir.UTF8_CODE,None)
	if len(sys.argv) < 6:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), int(sys.argv[5].decode("utf-8")))
	nlpir.Exit()
