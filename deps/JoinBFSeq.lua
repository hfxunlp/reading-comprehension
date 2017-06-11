local JoinBFSeq, parent = torch.class('nn.JoinBFSeq', 'nn.Module')

function JoinBFSeq:__init()
	parent.__init(self)
end

function JoinBFSeq:updateOutput(input)
	local f, i = unpack(input)
	local isize = i:size()
	local seql = isize[1]
	isize[1] = seql + 2
	if not self.output:isSize(isize) then
		self.output:resize(isize)
	end
	self.output[1]:copy(f)
	self.output:narrow(1, 2, seql):copy(i)
	self.output[-1]:copy(f)
	return self.output
end

function JoinBFSeq:updateGradInput(input, gradOutput)
	gradOutput[1]:add(gradOutput[-1])
	self.gradInput = {gradOutput[1], gradOutput:narrow(1, 2, gradOutput:size(1) - 2)}
	return self.gradInput
end

function JoinBFSeq:clearState()
	return parent.clearState(self)
end