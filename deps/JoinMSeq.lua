local JoinMSeq, parent = torch.class('nn.JoinMSeq', 'nn.Module')

function JoinMSeq:__init()
	parent.__init(self)
end

function JoinMSeq:updateOutput(input)
	local f, i, fe = unpack(input)
	local isize = input:size()
	local seql = isize[1]
	isize[1] = seql + 2
	if not self.output:isSize(isize) then
		self.output:resize(isize)
	end
	self.output[1]:copy(f)
	self.output:narrow(1, 2, seql):copy(i)
	self.output[-1]:copy(fe)
	return self.output
end

function JoinMSeq:updateGradInput(input, gradOutput)
	self.gradInput = {gradOutput[1], gradOutput:narrow(1, 2, gradOutput:size(1) - 2), gradOutput[-1]}
	return self.gradInput
end

function JoinMSeq:clearState()
	return parent.clearState(self)
end