local TableContainer, parent = torch.class('nn.TableContainer', 'nn.SequenceContainer')

function TableContainer:__init(module, shareModule)
	parent.__init(self, module, shareModule)
end

function TableContainer:updateOutput(input)
	if self.shareModule or not self.train then
		for _, v in ipairs(input) do
			self.output[_] = self:net(_):updateOutput(v):clone()
		end
	else
		for _, v in ipairs(input) do
			self.output[_] = self:net(_):updateOutput(v)
		end
	end
	for _ = #input + 1, #self.output do
		self.output[_] = nil
	end
	return self.output
end

function TableContainer:updateGradInput(input, gradOutput)
	for _ = #input + 1, #self.gradInput do
		self.gradInput[_] = nil
	end
	if self.shareModule then
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):updateGradInput(input[_], v):clone()
		end
	else
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):updateGradInput(input[_], v)
		end
	end
	return self.gradInput
end

function TableContainer:accGradParameters(input, gradOutput, scale)
	for _, v in ipairs(gradOutput) do
		self:net(_):accGradParameters(input[_], v, scale)
	end
end

function TableContainer:backward(input, gradOutput, scale)
	for _ = #input + 1, #self.gradInput do
		self.gradInput[_] = nil
	end
	if self.shareModule then
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):backward(input[_], v, scale):clone()
		end
	else
		for _, v in ipairs(gradOutput) do
			self.gradInput[_] = self:net(_):backward(input[_], v, scale)
		end
	end
	return self.gradInput
end
