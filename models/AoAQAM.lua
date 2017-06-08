require "nngraph"

return function (osize, hsize, nlayer)
	local function onehalfsize(sizein)
		local rs = math.ceil(sizein * 1.5)
		if rs % 2 == 1 then
			rs = rs + 1
		end
		return rs
	end
	require "deps.vecLookup"
	local qvm = nn.vecLookup(wvec)
	local pvm = qvm:clone('weight', 'gradWeight', 'bias', 'gradBias')
	local isize = wvec:size(2)
	hsize = hsize or isize--onehalfsize(isize)
	nlayer = nlayer or 1
	local buildEncoder = cudnn.BGRU
	local PEnc = buildEncoder(isize, hsize, nlayer)
	local QEnc = PEnc:clone('weight', 'gradWeight', 'bias', 'gradBias')
	local inputp = nn.Identity()()
	local vp = pvm(inputp)
	local vq = qvm()
	local p=PEnc(vp)
	local q=QEnc(vq)
	require "deps.CScore"
	require "deps.AoA"
	require "deps.fColl"
	local output = nn.CScore()({p, q})
	output = nn.AoA()(output)
	output = nn.fColl()({inputp, output})
	output = nn.Log()(output)
	return nn.gModule({inputp, vq}, {output})
end
