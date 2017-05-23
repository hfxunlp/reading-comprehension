#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def handle(srctf, rsf):
	cache = []
	with open(rsf, "w") as fwrt:
		with open(srctf) as frd:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						fwrt.write("\n".join(cache).encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
						cache = []
					else:
						tmp = tmp[tmp.find("|||")+4:]
						if tmp.find("XXXXX") != -1:
							tmp = tmp.replace("XXXXX", "")
						if tmp.find("?") != -1:
							tmp = tmp.replace("?", u"？")
						cache.append(tmp)

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
