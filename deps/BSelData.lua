local BSelData, parent = torch.class('nn.BSelData', 'nn.Module')

function BSelData:__init(odrop)
	parent.__init(self)
	self.odrop = odrop or 1
	self.ndrop = self.odrop * 2
end

function BSelData:updateOutput(input)
	self.output = input:narrow(1, self.odrop + 1, input:size(1) - self.ndrop)
	return self.output
end

function BSelData:updateGradInput(input, gradOutput)
	if not self.gradInput:isSize(input:size()) then
		self.gradInput:resizeAs(input)
	end
	local seql = input:size(1)
	self.gradInput:narrow(1, 1, self.odrop):zero()
	self.gradInput:narrow(1, seql - self.odrop + 1, self.odrop):zero()
	self.gradInput:narrow(1, self.odrop + 1, seql - self.ndrop):copy(gradOutput)
	return self.gradInput
end

function BSelData:clearState()
	return parent.clearState(self)
end