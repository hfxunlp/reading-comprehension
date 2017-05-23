require "cudnn"

return function (isize, osize, nlayers, fullout)
	if not fullout then
		return nn.Sequential()
			:add(cudnn.GRU(isize, osize or isize, nlayers or 1))
			:add(nn.Select(1, -1))
	else
		return cudnn.GRU(isize, osize or isize, nlayers or 1)
	end
end
