require "nngraph"

function buildQAM(osize, hsize, cisize, nlayer)
	require "deps.vecLookup"
	require "deps.TableContainer"
	local qvm = nn.vecLookup(wvec)
	local pvm = nn.TableContainer(qvm:clone('weight', 'gradWeight', 'bias', 'gradBias'), true)
	local isize = wvec:size(2)
	hsize = hsize or isize
	cisize = cisize or hsize
	nlayer = nlayer or 1
	require "deps.SequenceContainer"
	require "models.CFHiQATagger"
	local buildEncoder = require "deps.fgru"
	
	local SentEnc = buildEncoder(isize, hsize, nlayer, true)
	local PEnc = buildEncoder(hsize, cisize, nlayer)
	local clsm = nn.Linear(isize + hsize * 2 + cisize + isize, osize)
	local corem = nn.CFHiQATagger(SentEnc, PEnc, clsm, true)
	--local inputp = nn.Identity()()
	--local inputq = nn.Identity()()
	local vp = pvm()--(inputp)
	local vq = qvm()--(inputq)
	buildEncoder = require "deps.gru"
	local QEnc = buildEncoder(isize, isize, nlayer)
	local qfeat = QEnc(vq)
	local output = corem({vp, qfeat})
	--return nn.gModule({inputp, inputq}, {output})
	return nn.gModule({vp, vq}, {output})
end