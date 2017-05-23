local ExpanData, parent = torch.class('nn.ExpanData', 'nn.Module')

function ExpanData:__init()
	parent.__init(self)
end

function ExpanData:updateOutput(input)
	local stdv, uv = unpack(input)
	local usize = uv:size()
	local rv = uv:reshape(1, usize[1], usize[2]):expandAs(stdv)
	self.output = {stdv, rv}
end

function ExpanData:updateGradInput(input, gradOutput)
	local _gstd, _ged = unpack(gradOutput)
	self.gradInput = {_gstd, _ged:sum(1):squeeze(1)}
	return self.gradInput
end

function ExpanData:clearState()
	return parent.clearState(self)
end