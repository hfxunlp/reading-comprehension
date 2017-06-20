require "nn"
require "cutorch"
require "cunn"
require "cudnn"

local function getonn()
	--local lmod = loadObject(cntrain).module
	require "nn"
	require "nn.Decorator"
	require "dpnn"
	local buildQAM=require "models.ne_2GAoAQAM"
	local _rm=buildQAM(1)
	_rm=nil
	wvec = nil
	local lmod = torch.load(cntrain).modules[1]
	return lmod
end

local function getnnn()

	local buildQAM=require "models.ne_2GAoAQAM"
	--return buildQAM(1, nil, nil, nil, nil, 0.2)
	return buildQAM(1)
end

function getnn()
	if cntrain then
		return getonn()
	else
		return getnnn()
	end
end

function getcrit()
	return nn.ClassNLLCriterion();
	--return nn.MultiMarginCriterion();
end

local function setvec(net, value)
	net:apply(function(m)
		if torch.isTypeOf(m, 'nn.vecLookup') then
			m.updatevec = value
		end
	end)
end

function upvec(net)
	setvec(net, true)
end

function keepvec(net)
	setvec(net, nil)
end
