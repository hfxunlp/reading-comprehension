require "nngraph"
require "deps.vecLookup"
require "deps.CScore"
require "deps.AoA"
require "deps.fColl"

return function (osize, hsize, nlayer)
	local function mksize(sizein, vl)
		local rs = math.ceil(sizein * vl)
		if rs % 2 == 1 then
			rs = rs + 1
		end
		return rs
	end
	local qvm = nn.vecLookup(wvec)
	local pvm = qvm:clone('weight', 'gradWeight', 'bias', 'gradBias')
	local isize = wvec:size(2)
	hsize = hsize or isize--mksize(isize, 0.5)
	nlayer = nlayer or 1
	local buildEncoder = cudnn.BGRU
	local QEnc = buildEncoder(isize, hsize/2, nlayer)
	buildEncoder = require "deps.fbgru"
	local PEnc = buildEncoder(isize, hsize, nlayer)
	buildEncoder = require "deps.grcnn"
	local QFeatN = buildEncoder(nil, nil, true)
	local inputp = nn.Identity()()
	local vp = pvm(inputp)
	local vq = qvm()
	local qfeat = QFeatN(vq)
	local p=PEnc({qfeat, vp})
	local q=QEnc(vq)
	local output = nn.CScore()({p, q})
	output = nn.AoA()(output)
	output = nn.fColl()({inputp, output})
	output = nn.Log()(output)
	return nn.gModule({inputp, vq}, {output})
end
