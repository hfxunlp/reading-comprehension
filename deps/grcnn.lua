require "nn"
require "nngraph"
require "cudnn"
require "deps.vecLookup"
require "deps.JoinFSeq"

return function (nlayer, pdrop, stdbuild)
	local vsize = wvec:size(2)
	nlayer = nlayer or 1
	local input = nn.Identity()()
	local rgate = cudnn.GRU(vsize, vsize, nlayer, nil, pdrop)(input)-- better to find a way which can get bi-directional information
	local nrgate = nn.Sigmoid()(rgate)
	local appr = nn.CMulTable()({input, nrgate})
	local ws = cudnn.GRU(vsize, vsize, nlayer, nil, pdrop)(appr)
	local w_hat = nn.Select(1, -1)(ws)
	local newseq = nn.JoinFSeq()({w_hat, input})
	local zv = cudnn.GRU(vsize, vsize, nlayer, nil, pdrop)(newseq)
	local nzv
	if stdbuild then
		nzv = nn.Transpose({1, 3})(nn.Bottle(nn.SoftMax())(nn.Transpose({1, 3})(zv)))
	else
		nzv = nn.Bottle(nn.SoftMax())(zv)-- transpose {1, 3} for standard module
	end
	local cseq = nn.CMulTable()({nzv, newseq})
	local output = nn.Sum(1)(cseq)
	return nn.gModule({input}, {output})
end
