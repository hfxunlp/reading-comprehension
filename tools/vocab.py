#encoding: utf-8

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def handle(srctf, rsf):
	rsd = {}
	with open(srctf) as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if not tmp.startswith("<qid_"):
					tmp = tmp[tmp.find("|||")+4:].split()
					for tmpu in tmp:
						tt = tmpu.strip()
						if tt:
							rsd[tt] = rsd.get(tt, 0) + 1
	tmp = {}
	for k, v in rsd.iteritems():
		if not v in tmp:
			tmp[v] = [k]
		else:
			tmp[v].append(k)
	rsd = tmp.keys()
	rsd.sort(reverse=True)
	with open(rsf, "w") as f:
		for ru in rsd:
			tt = [str(ru)]
			tt.extend(tmp[ru])
			tt = " ".join(tt)
			f.write(tt.encode("utf-8"))
			f.write("\n".encode("utf-8"))

if __name__=="__main__":
	handle(sys.argv[1].decode("gbk"), sys.argv[2].decode("gbk"))
