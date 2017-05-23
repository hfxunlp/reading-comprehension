local DPrint, parent = torch.class('nn.DPrint', 'nn.Module')

function DPrint:__init(rpd)
	parent.__init(self)
	self.rpd = rpd
end

function DPrint:updateOutput(input)
	self.output = input
	print(self.rpd.." forward")
	print(self.output)
	return self.output
end

function DPrint:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput
	print(self.rpd.." backward")
	print(self.gradInput)
	return self.gradInput
end

function DPrint:clearState()
	return parent.clearState(self)
end