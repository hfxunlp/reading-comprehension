require "nngraph"

return function (osize, hsize, cisize, nlayer, hidsize, pdrop)
	local function halfsize(sizein)
		local rs = math.ceil(sizein / 2)
		if rs % 2 == 1 then
			rs = rs + 1
		end
		return rs
	end
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
	require "models.NICPFullTagger"
	local buildEncoder = require "deps.fgru"
	local SentEnc = buildEncoder(isize, hsize * 2, nlayer, true, pdrop)
	local PEnc = buildEncoder(hsize, cisize * 2, nlayer, true, pdrop)
	require "deps.PScore"
	require "deps.AoA"
	local clsm_isize = isize + hsize * 2 + cisize * 2 + isize
	local clsm_hsize = hidsize or halfsize(clsm_isize)
	local clsm_core = nn.Sequential()
		:add(nn.Linear(clsm_isize, clsm_hsize))
		:add(nn.Tanh())
		:add(nn.Linear(clsm_hsize, osize, false))
	local clsm = nn.Sequential()
		:add(nn.PScore(clsm_core))
		:add(nn.AoA())
	local corem = nn.NICPFullTagger(SentEnc, PEnc, clsm, true, isize)
	local inputp = nn.Identity()()
	local inputq = nn.Identity()()
	local vp = pvm(inputp)--()
	local vq = qvm(inputq)--()
	buildEncoder = require "deps.gru"
	local QEnc = buildEncoder(isize, isize * 2, nlayer, true, pdrop)
	local qfeat = QEnc(vq)
	local _output = corem({vp, qfeat})
	require "deps.Coll"
	local output = nn.Coll()({inputp, _output})
	output = nn.Log()(output)
	return nn.gModule({inputp, inputq}, {output})
	--return nn.gModule({vp, vq}, {output})
end
