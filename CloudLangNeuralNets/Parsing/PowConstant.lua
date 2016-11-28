require('nn')

local PowConstant, parent = torch.class('nn.PowConstant', 'nn.Module')

function PowConstant:__init(constant_scalar)
  parent.__init(self)
  assert(type(constant_scalar) == 'number', 'input is not scalar!')
  assert(constant_scalar > 1, 'For now, only greater-than-one constants are supported.')
  self.constant_scalar = constant_scalar
end

function PowConstant:updateOutput(input)
  self.output:resizeAs(input)
  self.output:copy(input)
  self.output:pow(self.constant_scalar)
  return self.output
end 

function PowConstant:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  self.gradInput:mul(self.constant_scalar)
  self.gradInput:cmul(torch.pow(input, self.constant_scalar-1))
  return self.gradInput
end
