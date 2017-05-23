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

def mapline(lind, mapd):
	ind = lind.find("|||") + 4
	rs = [lind[:ind - 1]]
	for linu in lind[ind:].split(" "):
		tt = linu.strip()
		if tt:
			rs.append(mapd.get(tt, "1"))
	return " ".join(rs)

def handle(srctf, rsf, mapf, minkeep = 5):
	mapd = ldmap(mapf, minkeep)
	with open(srctf) as frd:
		with open(rsf, "w") as fwrt:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					tmp = mapline(tmp, mapd)
					fwrt.write(tmp.encode("utf-8"))
					fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	if len(sys.argv) < 5:
		handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
	else:
		handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"), int(sys.argv[4].decode("gbk")))
