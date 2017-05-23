#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json

def handle(srcif, srctf, rsf):
	with open(rsf, "w") as fwrt:
		with open(srcif) as frdi:
			with open(srctf) as frdt:
				id = []
				td = []
				for il, tl in zip(frdi, frdt):
					ils = il.strip()
					tls = tl.strip()
					if ils:
						ils = ils.decode("utf-8")
						tls = tls.decode("utf-8")
						if ils.startswith("<qid_"):
							tmp = [id, [int(u) for u in ils[ils.find("|||") + 4:].split()], td]
							tmp = json.dumps(tmp)
							fwrt.write(tmp)
							fwrt.write("\n")
							id = []
							td = []
						else:
							ind = ils.find("|||") + 4
							id.append([int(u) for u in ils[ind:].split()])
							ind = tls.find("|||") + 4
							td.append([int(u) for u in tls[ind:].split()])

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"), sys.argv[3].decode("gbk"))
