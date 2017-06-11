function getoptim()
	local optim = require "optim"
	--return optim.adam
	return optim.sgd
end
