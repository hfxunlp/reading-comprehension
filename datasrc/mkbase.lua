local json = require("dkjson")
local tds = require("tds")

local vsize = 50

function ldvec(fsrc,vsize)
	local file=io.open(fsrc)
	local num=file:read("*n")
	local rs=tds.Vec()
	local curd=1
	while num do
		local t={}
		for i=1,vsize do
			table.insert(t,num)
			num=file:read("*n")
		end
		rs[curd]=torch.FloatTensor(t)
		curd = curd + 1
	end
	file:close()
	local ts=torch.FloatTensor(#rs, vsize)
	for i=1,#rs do
		ts[i]:copy(rs[i])
	end
	return ts
end

function conjson(fname)
	local function flatten(tc)
		local rs={}
		for _, t in ipairs(tc) do
			for __, v in ipairs(t:reshape(t:size(1)):totable()) do
				table.insert(rs, v)
			end
		end
		return torch.IntTensor(rs):reshape(#rs, 1)
	end
	local function convt(tin)
		local rsv=tds.Vec()
		for _,v in ipairs(tin) do
			rsv[_]=torch.IntTensor(v):reshape(#v, 1)
		end
		return rsv
	end
	local function fconvt(tin)
		local rs={}
		for _, v in ipairs(tin) do
			for __, r in ipairs(v) do
				table.insert(rs, r)
			end
		end
		rs=torch.IntTensor(rs)
		--rs[rs:eq(0)]=-1 --2 for Multi-Margin, -1 for Margin
		return rs
	end
	local function getarg(x, tin)
		local vocab = {}
		local curid = 1
		local curwd = 1
		local fscore = tin:reshape(tin:size(1)):totable()
		local fnd = nil
		for _, sent in ipairs(x) do
			for __, wd in ipairs(sent:reshape(sent:size(1)):totable()) do
				if not vocab[wd] then
					vocab[wd] = curid
					curid = curid + 1
				end
				if (fscore[curwd] or 1) == 1 then
					fnd = wd
					break
				else
					curwd = curwd + 1
				end
			end
			if fnd then
				break
			end
		end
		return torch.IntTensor({vocab[fnd] or (curid - 1)})
	end
	local file=io.open(fname)
	local rs=tds.Vec()
	local lind=file:read("*l")
	local curd=1
	while lind do
		local data=json.decode(lind)
		local id, qd, td=unpack(data)
		id = convt(id)
		td = getarg(id, fconvt(td))
		id = flatten(id)
		rs[curd]=tds.Vec(id, torch.IntTensor(qd):reshape(#qd, 1), td)
		lind=file:read("*l")
		curd=curd+1
	end
	file:close()
	return rs
end

--torch.save("debug.asc",conjson("duse/valid.data"))
--torch.save("debug.asc", tds.Vec(conjson("duse/valid.data"), conjson("duse/valid.data"), ldvec("duse/wvec_50.txt", 50)), 'binary', false)
torch.save(vsize.."kbdata.asc", tds.Vec(conjson("duse/train.data"), conjson("duse/valid.data"), ldvec("duse/kwvec_"..vsize..".txt", vsize)), 'binary', false)
