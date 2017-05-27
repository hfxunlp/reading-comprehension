local AoA, parent = torch.class('nn.AoA', 'nn.Container')

function AoA:__init()
	parent.__init(self)
	self.pnorm = nn.SoftMax()
	self.qnorm = nn.SoftMax()
	self:add(self.pnorm)
	self:add(self.qnorm)
	self._pout = torch.Tensor()
	self._qout = torch.Tensor()end

function AoA:updateOutput(input)
	local _qnorm = self.qnorm:updateOutput(input):sum(1)
	_qnorm:div(input:size(1))
	self._qout = _qnorm:expandAs(input)
	self._pout = self.pnorm:updateOutput(input:t()):t()
	self.output = torch.cmul(self._pout, self._qout):sum(2)
	return self.output
end

function AoA:updateGradInput(input, gradOutput)
	local _grad = gradOutput:expandAs(input)
	local grad = torch.cmul(_grad, self._qout)
	self.gradInput = self.pnorm:updateGradInput(input:t(), grad:t()):t()
	grad = torch.cmul(_grad, self._pout):sum(1)
	grad:div(input:size(1))
	grad = grad:expandAs(input)
	self.gradInput:add(self.qnorm:updateGradInput(input, grad))
	return self.gradInput
end

function AoA:clearState()
	self._pout:set()
	self._qout:set()
	return parent.clearState(self)
end