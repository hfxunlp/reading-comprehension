local AoA, parent = torch.class('nn.AoA', 'nn.Container')

function AoA:__init()
	parent.__init(self)
	self.pnorm = nn.SoftMax()
	self.qnorm = nn.SoftMax()
	self:add(self.pnorm)
	self:add(self.qnorm)
	self._pout = torch.Tensor()
	self._qout = torch.Tensor()
	self.gradp = torch.Tensor()
	self.gradq = torch.Tensor()
end

function AoA:updateOutput(input)
	local _qnorm = self.qnorm:updateOutput(input):sum(1)
	_qnorm:div(input:size(1))
	self._qout = _qnorm:expandAs(input)
	self._pout = self.pnorm:updateOutput(input:t()):t()
	self.output = torch.cmul(self._pout, self._qout):sum(2)
	self.doupgi = nil
	return self.output
end

function AoA:updateGradInput(input, gradOutput)
	if not self.doupgi then
		local _grad = gradOutput:expandAs(input)
		self.gradp = torch.cmul(_grad, self._qout)
		self.gradInput = self.pnorm:updateGradInput(input:t(), self.gradp:t()):t()
		local _gradq = torch.cmul(_grad, self._pout):sum(1)
		_gradq:div(input:size(1))
		self.gradq = _gradq:expandAs(input)
		self.gradInput:add(self.qnorm:updateGradInput(input, self.gradq))
		self.doupgi = true
	end
	return self.gradInput
end

function AoA:accGradParameters(input, gradOutput, scale)
	if not self.doupgi then
		self.updateGradInput(input, gradOutput)
	end
	self.pnorm:accGradParameters(input:t(), self.gradp:t(), scale)
	self.qnorm:accGradParameters(input, self.gradq, scale)
end

function AoA:backward(input, gradOutput, scale)
	local _grad = gradOutput:expandAs(input)
	self.gradp = torch.cmul(_grad, self._qout)
	self.gradInput = self.pnorm:backward(input:t(), self.gradp:t(), scale):t()
	local _gradq = torch.cmul(_grad, self._pout):sum(1)
	_gradq:div(input:size(1))
	self.gradq = _gradq:expandAs(input)
	self.gradInput:add(self.qnorm:backward(input, self.gradq, scale))
	return self.gradInput
end

function AoA:clearState()
	self._pout:set()
	self._qout:set()
	self.gradp:set()
	self.gradq:set()
	self.doupgi = nil
	return parent.clearState(self)
end