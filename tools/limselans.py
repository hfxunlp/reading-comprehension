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

def getans_mwd(pas, scl):
	tmp = scl.strip()
	tmp = tmp.decode("utf-8")
	tmp = [float(tmpu) for tmpu in tmp.split()]
	s = {}
	curid = 0
	for tmpu in tmp:
		s[curid] = tmpu
		curid += 1
	src = set()
	cache = set()
	for sent in pas:
		src |= set(sent)
		tmp = segline("".join(sent).replace(" ","")).split(" ")
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
	curid = 0
	for sent in pas:
		for wd in sent:
			_s = s.get(curid, -99999.99)
			if wd in rsd:
				if rsd[wd] < _s:
					rsd[wd] = _s
			else:
				rsd[wd] = _s
			curid += 1
	rsw = rsd.keys()[0]
	mscore = rsd[rsw]
	lim = False
	for wd in alw:
		if wd in rsw:
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

def getans_cwd(pas, scl):
	tmp = scl.strip()
	tmp = tmp.decode("utf-8")
	tmp = [float(tmpu) for tmpu in tmp.split()]
	s = {}
	curid = 0
	for tmpu in tmp:
		s[curid] = tmpu
		curid += 1
	src = set()
	cache = set()
	for sent in pas:
		src |= set(sent)
		tmp = segline("".join(sent).replace(" ","")).split(" ")
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
	curid = 0
	for sent in pas:
		for wd in sent:
			rsd[wd] = rsd.get(wd, 0.0) + s.get(curid, -99999.99)
			curid += 1
	rsw = rsd.keys()[0]
	mscore = rsd[rsw]
	lim = False
	for wd in alw:
		if wd in rsw:
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
