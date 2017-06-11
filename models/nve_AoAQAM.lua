require "nngraph"

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
	local pvm = nn.vecLookup(wvec)
	local isize = wvec:size(2)
	hsize = hsize or isize--mksize(isize, 0.5)
	nlayer = nlayer or 1
	local buildEncoder = cudnn.BGRU
	local PEnc = buildEncoder(isize, hsize/2, nlayer)
	local QEnc = buildEncoder(isize, hsize/2, nlayer)
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
