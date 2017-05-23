require "nn"
require "models.QAM"
require "cutorch"
require "cunn"
require "cudnn"

local function getonn()
	wvec = nil
	--local lmod = loadObject("modrs/nnmod.asc").module
	local lmod = torch.load("modrs/nnmod.asc").module
	return lmod
end

local function getnnn()

	require "models.QAM"
	return buildQAM(1)
end

function getnn()
	--return getonn()
	return getnnn()
end

function getcrit()
	return nn.MarginCriterion();
end
