require "nngraph"

return function()
	local seq = nn.Transpose({1, 2})()--bsize slen vsize
	local std = nn.Identity()()--bsize vsize
	local _a = nn.Squeeze(3)(nn.MM()({seq, nn.ADim(3)(std)}))--bsize slen
	local a = nn.ADim(2)(nn.SoftMax()(_a))--bsize 1 slen
	local _q = nn.Squeeze(2)(nn.MM()({a, seq}))--bsize vsize
	local output = nn.CMulTable()({_q, std})--bsize vsize
	return nn.gModule({seq, std}, {output})
end