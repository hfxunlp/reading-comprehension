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

def checksame(str1, str2):
	if str1 == str2:
		return "1"
	else:
		return "0"

def clearl(lin):
	rs = []
	for lu in lin:
		tt = lu.strip()
		if tt:
			rs.append(tt)
	return rs

def buildsame(strin, ind, ans):
	rs = [strin[:ind - 1]]
	rs.extend([checksame(su, ans) for su in clearl(strin[ind:].split(" "))])
	return " ".join(rs)

def mapline(lind, ind, mapd):
	rs = [lind[:ind - 1]]
	for linu in lind[ind:].split(" "):
		tt = linu.strip()
		if tt:
			rs.append(mapd.get(tt, "1"))
	return " ".join(rs)

def handle(srctf, srcrf, mapf, rsf, minkeep = 5):
	ans = ldans(srcrf)
	mapd = ldmap(mapf, minkeep)
	cache = []
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						curans = ans[tmp[:tmp.find(">") + 1]]
						rs = []
						for cu in cache:
							ind = cu.find("|||") + 4
							rs.append(buildsame(cu, ind, curans))
						ind = tmp.find("|||") + 4
						rs.append(mapline(tmp, ind, mapd))
						tmp = "\n".join(rs)
						fwrt.write(tmp.encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						cache = []
					else:
						cache.append(tmp)

if __name__=="__main__":
	if len(sys.argv) < 6:
		handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"), sys.argv[4].decode("gbk"))
	else:
		handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"), sys.argv[4].decode("gbk"), int(sys.argv[5].decode("gbk")))
