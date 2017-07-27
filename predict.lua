local tif = arg[1] or "datasrc/duse/valid.data" -- test input, json format
local rsf = arg[2] or "test/aoanscore.txt" -- result score file
local modf = arg[3] or "modrs/170614_ne_2aoabase_088sgd_0001_v50_h50/devnnmod1.asc" -- model file

torch.setdefaulttensortype('torch.FloatTensor')

local tds = require("tds")
local json = require("dkjson")

local function ldjson(fname)
	local function convt(tin)
		local rsv=tds.Vec()
		for _,v in ipairs(tin) do
			rsv[_]=torch.IntTensor(v):reshape(#v, 1)
		end
		return rsv
	end
	local file=io.open(fname)
	local rs=tds.Vec()
	local lind=file:read("*l")
	local curd=1
	while lind do
		local data=json.decode(lind)
		local id, qd=unpack(data)
		rs[curd]=tds.Vec(convt(id), torch.IntTensor(qd):reshape(#qd, 1))
		lind=file:read("*l")
		curd=curd+1
	end
	file:close()
	return rs
end

local function mkcudaLong(din)
	local rs = {}
	for _, v in ipairs(din) do
		table.insert(rs, v:cudaLong())
	end
	return rs
end

require "nn"
require "cutorch"
require "cunn"
require "cudnn"

require "nn.Decorator"
require "dpnn"

require "nngraph"

require "deps.vecLookup"
require "deps.JoinFSeq"
require "deps.JoinBFSeq"
require "deps.SelData"
require "deps.BSelData"
require "deps.TableContainer"
require "deps.SequenceContainer"
require "deps.Coll"
require "deps.PScore"
require "deps.AoA"
require "models.NICPFullTagger"
require "deps.CScore"
require "deps.fColl"

local tmod_full = torch.load(modf)
local tmod = tmod_full.modules[1]
tmod:evaluate()
local tdata = ldjson(tif)

local file = io.open(rsf, "w")

local function flatten(tc)
	local rs={}
	for _, t in ipairs(tc) do
		for __, v in ipairs(t:reshape(t:size(1)):totable()) do
			table.insert(rs, v)
		end
	end
	return torch.IntTensor(rs):reshape(#rs, 1)
end

for _, dtest in ipairs(tdata) do
	--local rs = tmod:updateOutput({mkcudaLong(dtest[1]), dtest[2]:cudaLong()}):float()
	local rs = tmod:updateOutput({flatten(dtest[1]):cudaLong(), dtest[2]:cudaLong()}):float()
	local rs = rs:reshape(rs:size(1)):totable()
	local wrs = {}
	for __, v in ipairs(rs) do
		table.insert(wrs, tostring(v))
	end
	rs = table.concat(wrs, " ")
	file:write(rs)
	file:write("\n")
end

file:flush()

file:close()
