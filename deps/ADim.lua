local ADim, parent = torch.class('nn.ADim', 'nn.Module')

function ADim:__init(dim)
	parent.__init(self)
	self.dim = dim
end

function ADim:updateOutput(input)
	local idim = input:size():totable()
	local adim = self.dim - #idim
	if adim > 0 then
		for _ = 1, adim do
			table.insert(idim, 1)
		end
	else
		table.insert(idim, self.dim, 1)
	end
	self.output:set(input:view(torch.LongStorage(idim)))
	return self.output
end

function ADim:updateGradInput(input, gradOutput)
	self.gradInput:set(gradOutput:view(input:size()))
	return self.gradInput
end