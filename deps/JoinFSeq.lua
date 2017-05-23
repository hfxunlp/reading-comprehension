local JoinFSeq, parent = torch.class('nn.JoinFSeq', 'nn.Module')

function JoinFSeq:__init()
	parent.__init(self)
end

function JoinFSeq:updateOutput(input)
	local f, i = unpack(input)
	local isize = i:size()
	local seql = isize[1]
	isize[1] = seql + 1
	if not self.output:isSize(isize) then
		self.output:resize(isize)
	end
	self.output[1]:copy(f)
	self.output:narrow(1, 2, seql):copy(i)
	return self.output
end

function JoinFSeq:updateGradInput(input, gradOutput)
	self.gradInput = {gradOutput[1], gradOutput:narrow(1, 2, gradOutput:size(1) - 1)}
	return self.gradInput
end

function JoinFSeq:clearState()
	return parent.clearState(self)
end