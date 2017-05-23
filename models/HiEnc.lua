local HiEnc, parent = torch.class('nn.HiEnc', 'nn.SequenceContainer')

-- sentEnc and PEnc should not enable fullout
function HiEnc:__init(sentEnc, PEnc)
	parent.__init(self, sentEnc)
	self.PEnc = PEnc or sentEnc
	self:add(self.PEnc)
	self.cells = torch.Tensor()
end

-- input should be table which contains its sentences
function HiEnc:updateOutput(input)

	local seql = #input
	local _output = self:net(1):updateOutput(input[1])
	local _osize = _output:size()
	local stdsize = torch.LongStorage({seql, _osize[1], _osize[2]})
	if not self.cells:isSize(stdsize) then
		self.cells:resize(stdsize)
	end
	self.cells[1]:copy(_output)
	for _, sent in ipairs(input) do
		if _ > 1 then
			self.cells[_]:copy(self:net(_):updateOutput(sent))
		end
	end
	self.output = self.PEnc:updateOutput(self.cells)
	return self.output
end

function HiEnc:updateGradInput(input, gradOutput)
	self.gradCells = self.PEnc:updateGradInput(self.cells, gradOutput)
	self.gradInput = {}
	for _, v in ipairs(input) do
		table.insert(self:net(_):updateGradInput(v, self.gradCells[_]))
	end
	return self.gradInput
end

function HiEnc:accGradParameters(input, gradOutput, scale)
	self.PEnc:accGradParameters(self.cells, gradOutput, scale)
	for _, v in ipairs(input) do
		self:net(_):accGradParameters(v, self.gradCells[_], scale)
	end
end

function HiEnc:backward(input, gradOutput, scale)
	self.gradCells = self.PEnc:backward(self.cells, gradOutput, scale)
	self.gradInput = {}
	for _, v in ipairs(input) do
		table.insert(self:net(_):backward(v, self.gradCells[_], scale))
	end
	return self.gradInput
end

function HiEnc:clearState()
	self.cells:set()
	self.gradCells:set()
	return parent.clearState(self)
end