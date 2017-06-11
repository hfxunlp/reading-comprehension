require "cudnn"
require "nngraph"
require "rnn"
require "deps.JoinBFSeq"
require "deps.BSelData"

return function (isize, osize, nlayers, pdrop)
	osize = osize or isize
	size = osize/2
	nlayers = nlayers or 1
	local dinput = nn.Identity()()
	local finput = nn.Identity()()
	local input = nn.JoinBFSeq()({finput, dinput})
	local _output = cudnn.BGRU(isize, size, nlayers, nil, pdrop)(input)
	local output = nn.BSelData()(_output)
	return nn.gModule({finput, dinput}, {output})
end
