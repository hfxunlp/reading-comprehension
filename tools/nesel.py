#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
from pyltp import Postagger, NamedEntityRecognizer

def clearline(lin):
	rs = []
	for lu in lin:
		tmp = lu.strip()
		if tmp:
			rs.append(tmp.encode("utf-8"))
	return rs

def getent(srcl, fd, ptag, rec):
	wds = clearline(srcl)
	ptags = ptag.postag(wds)
	netags = list(rec.recognize(wds, ptags))
	nrs = set()
	for w, t in zip(wds, list(ptags)):
		if t.startswith("n"):
			w = w.decode("utf-8")
			if (w in fd) and (not w in nrs):
				nrs.add(w)
	ers = set()
	for w, n in zip(wds, netags):
		if n!="O":
			w = w.decode("utf-8")
			if (w in fd) and (not w in ers):
				ers.add(w)
	return nrs, ers

def getcand(pas, rsd, ptag, rec):
	nu = set()
	eu = set()
	for sent in pas:
		nc, ec = getent(sent, rsd, ptag, rec)
		nu |= nc
		eu |= ec
	if eu:
		return eu
	else:
		return nu

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

def getans_mwd(pas, scl, mapd, ptag, rec):
	tmp = scl.strip()
	tmp = tmp.decode("utf-8")
	s = [float(tmpu) for tmpu in tmp.split()]
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
	alw = getcand(pas, rsd, ptag, rec)
	rsw = rsd.keys()[0]
	lim = False
	if alw:
		for w in alw:
			if w!="XXXXX" and w in rsd:
				rsw = w
				print(w)
				lim = True
				break
	else:
		print("get no entity")
	mscore = rsd[rsw]
	for w, score in rsd.iteritems():
		if (not lim) or (w in alw):
			if score > mscore:
				mscore = score
				rsw = w
	return rsw, mscore

def getans(pas, scl, mapd, ptag, rec):
	w, s = getans_mwd(pas, scl, mapd, ptag, rec)
	return w

def handle(mapf, srcif, srctf, rsf, minkeep = 5):
	global postagger, recognizer
	rs = []
	cache = []
	curid = 0
	mapd = ldmap(mapf, minkeep)
	with open(srcif) as frdi:
		with open(srctf) as frdt:
			for line in frdi:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8", "ignore")
					if tmp.startswith("<qid_"):
						rs.append(" ".join(["<qid_"+str(curid)+">", "|||", getans(cache, frdt.readline(), mapd, postagger, recognizer)]))
						cache = []
						curid += 1
					else:
						tmp = tmp[tmp.find("|||")+4:]
						cache.append(tmp.split())
	if cache:
		rs.append(" ".join(["<qid_"+str(curid)+">", "|||", getans(cache, frdt.readline(), mapd, postagger, recognizer)]))
		cache = []
	rs = "\n".join(rs)
	with open(rsf, "w") as fwrt:
		fwrt.write(rs.encode("utf-8"))

if __name__=="__main__":
	ltpdata="/media/Storage/data/ltp_data/"
	postagger = Postagger()
	postagger.load(os.path.join(ltpdata, "pos.model"))
	recognizer = NamedEntityRecognizer()
	recognizer.load(os.path.join(ltpdata, "ner.model"))

	if len(sys.argv) < 6:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), sys.argv[4].decode("utf-8"), int(sys.argv[5].decode("utf-8")))
