require "cutorch"

local PDataContainer = torch.class('PDataContainer')

function PDataContainer:__init(dset, ndata)
	self.dset = dset
	self.ndata = ndata or #dset
end

local function iter(dset, curid)
	local function mkcudaLong(din)
		local rs = {}
		for _, v in ipairs(din) do
			table.insert(rs, v:cudaLong())
		end
		return rs
	end
	curid = curid + 1
	local data = dset[curid]
	if data then
		return curid, {mkcudaLong(data[1]), data[2]:cudaLong()}, data[3]:cudaLong()
	else
		return
	end
end

function PDataContainer:subiter()
	return iter, self.dset, 0
end
