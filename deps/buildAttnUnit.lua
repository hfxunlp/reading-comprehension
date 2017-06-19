require "nngraph"

return function()
	local seq = nn.Transpose({1, 2})()
	local std = nn.Identity()()
	local _a = nn.Squeeze(3)(nn.MM()({seq, nn.ADim(3)(std)}))
	local a = nn.ADim(2)(nn.SoftMax()(_a))
	local _q = nn.Squeeze(2)(nn.MM()({a, seq}))
	local output = nn.CMulTable()({_q, std})
	return nn.gModule({seq, std}, {output})
end