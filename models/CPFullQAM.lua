require "nngraph"

return function (osize, hsize, cisize, nlayer, hidsize)
	local function onehalfsize(sizein)
		local rs = math.ceil(sizein * 1.5)
		if rs % 2 == 1 then
			rs = rs + 1
		end
		return rs
	end
	require "deps.vecLookup"
	require "deps.TableContainer"
	local qvm = nn.vecLookup(wvec)
	local pvm = nn.TableContainer(qvm:clone('weight', 'gradWeight', 'bias', 'gradBias'), true)
	local isize = wvec:size(2)
	hsize = hsize or isize
	cisize = cisize or onehalfsize(hsize)
	nlayer = nlayer or 1
	require "deps.SequenceContainer"
	require "models.CPFullTagger"
	local buildEncoder = require "deps.fgru"
	local SentEnc = buildEncoder(isize, hsize * 2, nlayer, true)
	local PEnc = buildEncoder(hsize, cisize * 2, nlayer, true)
	require "deps.PScore"
	require "deps.AoA"
	local clsm = nn.Sequential()
		:add(nn.PScore(nn.Linear(isize * 2 + hsize * 2 + cisize * 2 + isize, osize, false)))
		:add(nn.AoA())
	local corem = nn.CPFullTagger(SentEnc, PEnc, clsm, true, isize)
	local inputp = nn.Identity()()
	local inputq = nn.Identity()()
	local vp = pvm(inputp)--()
	local vq = qvm(inputq)--()
	buildEncoder = require "deps.gru"
	local QEnc = buildEncoder(isize, isize * 2, nlayer, true)
	local qfeat = QEnc(vq)
	local _output = corem({vp, qfeat})
	require "deps.Coll"
	local output = nn.Coll()({inputp, _output})
	output = nn.Log()(output)
	return nn.gModule({inputp, inputq}, {output})
	--return nn.gModule({vp, vq}, {output})
end
