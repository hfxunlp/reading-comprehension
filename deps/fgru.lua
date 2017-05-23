require "cudnn"
require "deps.JoinFSeq"

return function (isize, osize, nlayers, fullout)
	if not fullout then
		return nn.Sequential()
			:add(nn.JoinFSeq())
			:add(cudnn.GRU(isize, osize or isize, nlayers or 1))
			:add(nn.Select(1, -1))
	else
		require "deps.SelData"
		return nn.Sequential()
			:add(nn.JoinFSeq())
			:add(cudnn.GRU(isize, osize or isize, nlayers or 1))
			:add(nn.SelData())
	end
end
