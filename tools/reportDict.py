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

def handle(srctf):
	print(len(ldmap(srctf))-1)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"))
