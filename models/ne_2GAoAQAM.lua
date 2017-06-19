require "nngraph"
require "deps.CScore"
require "deps.AoA"
require "deps.fColl"
require "deps.SequenceContainer"
require "deps.ADim"
require "deps.GA"

return function (osize, hsize, nlayer)
	local function mksize(sizein, vl)
		local rs = math.ceil(sizein * vl)
		if rs % 2 == 1 then
			rs = rs + 1
		end
		return rs
	end
	require "deps.vecLookup"
	local qvm = nn.vecLookup(wvec)
	local pvm = qvm:clone('weight', 'gradWeight', 'bias', 'gradBias')
	local isize = wvec:size(2)
	hsize = hsize or isize--mksize(isize, 0.5)
	nlayer = nlayer or 1
	local buildEncoder = cudnn.BGRU
	local PEnc1 = buildEncoder(isize, hsize/2, nlayer)
	local PEnc2 = buildEncoder(hsize, hsize/2, nlayer)
	local buildGA = require "deps.buildAttnUnit"
	local GA = nn.GA(buildGA())
	local QEnc1 = buildEncoder(isize, hsize/2, nlayer)
	local QEnc2 = buildEncoder(isize, hsize/2, nlayer)
	local inputp = nn.Identity()()
	local vp = pvm(inputp)
	local vq = qvm()
	local p1=PEnc1(vp)
	local q1=QEnc1(vq)
	local _p2i=GA({p1, q1})
	local p2=PEnc2(_p2i)
	local q2=QEnc2(vq)
	local output = nn.CScore()({p2, q2})
	output = nn.AoA()(output)
	output = nn.fColl()({inputp, output})
	output = nn.Log()(output)
	return nn.gModule({inputp, vq}, {output})
end
