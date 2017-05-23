local SelData, parent = torch.class('nn.SelData', 'nn.Module')

function SelData:__init(odrop)
	parent.__init(self)
	self.odrop = odrop or 1
end

function SelData:updateOutput(input)
	self.output = input:narrow(1, self.odrop + 1, input:size(1) - self.odrop)
	return self.output
end

function SelData:updateGradInput(input, gradOutput)
	if not self.gradInput:isSize(input:size()) then
		self.gradInput:resizeAs(input)
	end
	self.gradInput:narrow(1, 1, self.odrop):zero()
	self.gradInput:narrow(1, self.odrop + 1, input:size(1) - self.odrop):copy(gradOutput)
	return self.gradInput
end

function SelData:clearState()
	return parent.clearState(self)
end