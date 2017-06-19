local GA, parent = torch.class('nn.GA', 'nn.SequenceContainer')

function GA:__init(gaunit)
	parent.__init(self, gaunit)
	self.gradDoc = torch.Tensor()
	self.gradQ = torch.Tensor()
end

function GA:updateOutput(input)
	local doc, q = unpack(input)
	local _s = doc:size()
	if not self.output:isSize(_s) then
		self.output:resize(_s)
	end
	for _ = 1, doc:size(1) do
		self.output[_]:copy(self:net(_):updateOutput({q, doc[_]}))
	end
	return self.output
end

function GA:updateGradInput(input, gradOutput)
	local doc, q = unpack(input)
	local _s = doc:size()
	if not self.gradDoc:isSize(_s) then
		self.gradDoc:resize(_s)
	end
	_s = q:size()
	if not self.gradQ:isSize(_s) then
		self.gradQ:resize(_s):zero()
	end
	for  _ = 1, doc:size(1) do
		local _gq, _gdoc = unpack(self:net(_):updateGradInput({q, doc[_]}, gradOutput[_]))
		self.gradQ:add(_gq)
		self.gradDoc[_]:copy(_gdoc)
	end
	self.gradInput = {self.gradDoc, self.gradQ}
	return self.gradInput
end

function GA:accGradParameters(input, gradOutput, scale)
	local doc, q = unpack(input)
	for  _ = 1, doc:size(1) do
		self:net(_):accGradParameters({q, doc[_]}, gradOutput[_], scale)
	end
end

function GA:backward(input, gradOutput, scale)
	local doc, q = unpack(input)
	local _s = doc:size()
	if not self.gradDoc:isSize(_s) then
		self.gradDoc:resize(_s)
	end
	_s = q:size()
	if not self.gradQ:isSize(_s) then
		self.gradQ:resize(_s):zero()
	end
	for  _ = 1, doc:size(1) do
		local _gq, _gdoc = unpack(self:net(_):backward({q, doc[_]}, gradOutput[_], scale))
		self.gradQ:add(_gq)
		self.gradDoc[_]:copy(_gdoc)
	end
	self.gradInput = {self.gradDoc, self.gradQ}
	return self.gradInput
end

function GA:clearState()
	self.gradDoc:set()
	self.gradQ:set()
	return parent.clearState(self)
end