#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from math import sqrt

def ldvec(vecf):
	rsd = {}
	unkv = False
	with open(vecf) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				ind = tmp.find(" ")
				key = tmp[:ind]
				value = tuple(float(tmpu) for tmpu in tmp[ind+1:].split(" "))
				if key == "<unk>":
					unkv = value
				else:
					rsd[key] = value					
	if not unkv:
		unkv = (0.0 for i in xrange(len(rsd[rsd.keys()[0]])))
	return rsd, unkv

def ldcandidate(fname):
	rs = {}
	with open(fname) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8").split(" ")
				rs[tmp[0]] = tmp[1:]
	return rs

def add_vec(v1, v2):
	return tuple(v1u + v2u for v1u, v2u in zip(v1, v2))

def mul_vec(v1, v2):
	return tuple(v1u * v2u for v1u, v2u in zip(v1, v2))

def dot_vec(v1, v2):
	return sum_vec(mul_vec(v1, v2))

def sum_vec(vl):
	sum = 0
	for vu in vl:
		sum += vu
	return sum

def cos_vec(v1, v2):
	nv1 = dot_vec(v1, v1)
	nv2 = dot_vec(v2, v2)
	d = dot_vec(v1, v2)
	return d / sqrt(nv1 * nv2)

def norm_vec(vl):
	s = dot_vec(vl, vl)
	if s > 0:
		s = sqrt(s)
		return tuple(vu/s for vu in vl)
	else:
		return vl

def sentvec(lin, vd, unkv):
	rs = False
	for lu in lin:
		if not rs:
			rs = vd.get(lu, unkv)
		else:
			rs = add_vec(rs, vd.get(lu, unkv))
	return rs

def sentvecnounk(lin, vd):
	rs = False
	for lu in lin:
		if not rs:
			if lu in vd:
				rs = vd[lu]
		else:
			if lu in vd:
				rs = add_vec(rs, vd[lu])
	return rs

def g_class(svec, classes, vecd):
	rs = ""
	rscore = -1.1
	for k in classes:
		if k in vecd:
			curscore = cos_vec(vecd[k], svec)
			if curscore > rscore:
				rscore = curscore
				rs = k
	return rs, rscore

def getwindow(lin, wsize):
	curid = 0
	for lu in lin:
		if lu == "XXXXX":
			break
		curid += 1
	lind = curid - wsize
	rind = curid + wsize + 1
	if lind < 0:
		lind = 0
	l = len(lin)
	if rind > l:
		rind = l
	return lin[lind:rind]

def handle(srctf, srcf, vecf, rsf):
	vecs, unkv = ldvec(vecf)
	cand = ldcandidate(srcf)
	with open(srctf) as frd:
		with open(rsf,"w") as fwrt:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						ind = tmp.find(">")
						curid = tmp[:ind + 1]
						tmp, score = g_class(sentvecnounk(getwindow(tmp[tmp.find("|||")+4:].split(" "), 5), vecs), cand[curid], vecs)
						tmp = " ||| ".join((curid, tmp,))
						fwrt.write(tmp.encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"), sys.argv[4].decode("gbk"))
