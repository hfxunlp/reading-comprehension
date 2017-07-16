#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

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
						rs[wd] = str(curid)
						curid += 1
				else:
					break
	return rs

def clearl(lin):
	rs = []
	for lu in lin:
		tt = lu.strip()
		if tt:
			rs.append(tt)
	return rs

def buildsame(strin, ind):
	rs = [strin[:ind - 1]]
	rs.extend(["0" for su in clearl(strin[ind:].split(" "))])
	return " ".join(rs)

def mapline(lind, ind, mapd):
	rs = [lind[:ind - 1]]
	for linu in lind[ind:].split(" "):
		tt = linu.strip()
		if tt:
			rs.append(mapd.get(tt, "1"))
	return " ".join(rs)

def handle(srctf, mapf, rsf, minkeep = 5):
	mapd = ldmap(mapf, minkeep)
	cache = []
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						rs = []
						for cu in cache:
							ind = cu.find("|||") + 4
							rs.append(buildsame(cu, ind))
						ind = tmp.find("|||") + 4
						rs.append(mapline(tmp, ind, mapd))
						tmp = "\n".join(rs)
						fwrt.write(tmp.encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						cache = []
					else:
						cache.append(tmp)

if __name__=="__main__":
	if len(sys.argv) < 5:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"))
	else:
		handle(sys.argv[1].decode("utf-8"), sys.argv[2].decode("utf-8"), sys.argv[3].decode("utf-8"), int(sys.argv[4].decode("utf-8")))
