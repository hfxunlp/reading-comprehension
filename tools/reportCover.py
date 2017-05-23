#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def ldans(fname):
	rs = set()
	with open(fname) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				key, v = tmp.decode("utf-8").split(" ||| ")
				if not v in rs:
					rs.add(v)
	return rs

def handle(srctf, srcef):
	ans = ldans(srctf)
	total = 0
	cover = 0
	with open(srcef) as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				key, v = tmp.split(" ||| ")
				if v in ans:
					cover += 1
				total += 1
	print(float(cover)/total)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
