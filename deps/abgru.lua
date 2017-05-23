require "cudnn"
require "nngraph"
require "rnn"

return function (isize, osize, nlayers, fullout, size)
	osize = osize or isize
	size = size or osize
	nlayers = nlayers or 1
	local input = nn.Identity()()
	local rinput = nn.SeqReverseSequence(1)(input)
	local rinfo = cudnn.GRU(isize, size, nlayers)(rinput)
	local info = nn.SeqReverseSequence(1)(rinfo)
	local real_input = nn.ConcatTable(3)({input, info})
	local output = cudnn.GRU(isize + size, osize, nlayers)(real_input)
	if not fullout then
		output = nn.Select(1, -1)(output)
	end
	return nn.gModule({input}, {output})
end
