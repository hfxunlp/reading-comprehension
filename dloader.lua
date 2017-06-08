local tds = require "tds"

local _fulldata = torch.load("datasrc/192bdata.asc", 'binary', false)

local traindata, devdata

traindata, devdata, wvec = _fulldata[1], _fulldata[2], _fulldata[3]

wvec=wvec:float()

ntrain=#traindata
ndev=#devdata
nword=wvec:size(1)

if partrain and (partrain < ntrain) then
	for _ = partrain + 1, ntrain do
		traindata:remove()
	end
	ntrain = partrain
end

require "utils.DataContainer"

traind=traindata
devd=devdata

return {DataContainer(traindata), DataContainer(devdata)}
