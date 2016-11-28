require('dep.feature')
require('torch')
require('data')
require('util')

local vocabs = defaultdict(function() return Vocabulary() end)
local tokens = torch.LongTensor{
    {31, 9, 0, 2, 5}, -- root
    {29, 2, 3, 3, 2}, -- 1st word
    {7,  3, 1, 6, 4}, -- 2nd word
    {11, 2, 3, 5, 8}, -- 3rd word
    {13, 2, 4, 9, 7}, -- 4th word
}
local links = torch.LongTensor{
    {0, 0}, 
    {1, 7}, 
    {0, 0},
    {0, 0},
    {0, 0},
}
stack = {3, 1}
buffer = {4, 5}

function test_chen_manning()
    local cm = ChenManningFeatures()
    local x = torch.LongTensor(cm:num())
    cm(x, tokens, links, stack, buffer, vocabs) 
end

test_chen_manning()
print('OK')