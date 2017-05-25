local modf = "modrs/170524_gru_qf_coll/nnmod.asc" -- model file
local tif = "datasrc/duse/valid.data" -- test input, json format
local rsf = "test/canscore.txt" -- result score file

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
require "deps.SelData"
require "deps.TableContainer"
require "deps.SequenceContainer"
require "deps.MaxColl"
require "models.CFHiQATagger"

local tmod_full = torch.load(modf)
local tmod = tmod_full.modules[1]

local tdata = ldjson(tif)

local file = io.open(rsf, "w")

for _, dtest in ipairs(tdata) do
	local rs = tmod:updateOutput({mkcudaLong(dtest[1]), dtest[2]:cudaLong()}):float()
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
