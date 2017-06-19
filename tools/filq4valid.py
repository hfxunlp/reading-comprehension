#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def handle(srctf, rsf):
	with open(srctf) as frd:
		with open(rsf, "w") as fwrt:
			for line in frd:
				tmp = line.strip()
				if tmp:
					tmp = tmp.decode("utf-8")
					if tmp.startswith("<qid_"):
						fwrt.write(tmp.encode("utf-8"))
						fwrt.write("\n".encode("utf-8"))
					elif tmp.find("XXXXX")==-1:
							fwrt.write(tmp.encode("utf-8"))
							fwrt.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
