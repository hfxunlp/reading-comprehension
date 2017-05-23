require "cudnn"
require "nngraph"
require "rnn"
require "deps.JoinBFSeq"

return function (isize, osize, nlayers, fullout, hsize)
	osize = osize or isize
	size = size or osize
	nlayers = nlayers or 1
	local dinput = nn.Identity()()
	local finput = nn.Identity()()
	local input = nn.JoinBFSeq()({finput, dinput, finput})
	local rinput = nn.SeqReverseSequence(1)(input)
	local rinfo = cudnn.GRU(isize, size, nlayers)(rinput)
	local info = nn.SeqReverseSequence(1)(rinfo)
	local real_input = nn.JoinTable(3, 3)({input, info})
	local _output = cudnn.GRU(isize + size, osize, nlayers)(real_input)
	local output
	if not fullout then
		output = nn.Select(1, -1)(_output)
	else
		require "dep.BSelData"
		output = nn.BSelData()(_output)
	end
	return nn.gModule({finput, dinput}, {output})
end
