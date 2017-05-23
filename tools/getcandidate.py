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

def handle(srctf, rsf):
	cache = set()
	src = set()
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						fwrt.write(tmp[:tmp.find(">")+2].encode("utf-8"))
						wcache = []
						for cu in cache:
							if cu in src:
								wcache.append(cu)
						fwrt.write(" ".join(wcache).encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						cache = set()
						src = set()
					else:
						tmp = tmp[tmp.find("|||")+4:]
						if tmp.find("XXXXX") != -1:
							tmp = tmp.replace("XXXXX", "")
						src |= set(tmp.split(" "))
						tmp = segline(tmp).split(" ")
						for tmpu in tmp:
							ind = tmpu.rfind("/")
							wd, tag = tmpu[:ind], tmpu[ind+1:]
							if tag.startswith("n") and not wd in cache:
								cache.add(wd)

if __name__=="__main__":
	if nlpir.Init(nlpir.PACKAGE_DIR,nlpir.UTF8_CODE,None):
		handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
	else:
		print "Init error"
