require('dep.task')
require('dep.feature')
require('data')

local tokens = torch.LongTensor{
    {31, 9, 0, 0, 0}, -- root
    {29, 2, 3, 3, 3}, -- 1st word
    {7,  3, 1, 3, 3}, -- 2nd word
    {11, 2, 3, 3, 3}, -- 3rd word
    {13, 2, 4, 3, 3}, -- 4th word
}
local ds = {
    sents = torch.LongTensor{
        {1, 1, 5},
    },
    ['tokens'] = tokens
}
local input_vocabs = defaultdict(function() return Vocabulary() end)
local unlabeled_actions = make_actions({'dep', 'prop', 'v'}, input_vocabs.label)
-- print(unlabeled_actions)

function test_execute()
    local parser = ArcStandardParser(unlabeled_actions)
    parser:reset(tokens)
    parser:execute(unlabeled_actions.shift)
    parser:execute(unlabeled_actions.shift)
    parser:execute(unlabeled_actions.shift)
    parser:execute(unlabeled_actions.shift)
    -- print(parser.stack)
    parser:execute(unlabeled_actions.vocab:get_index('L(dep)'))
    -- print(parser.stack)
    parser:execute(unlabeled_actions.vocab:get_index('L(dep)'))
    -- print(parser.stack)
    parser:execute(unlabeled_actions.shift)
    -- print(parser.stack)
    parser:execute(unlabeled_actions.vocab:get_index('R(v)'))
    parser:execute(unlabeled_actions.vocab:get_index('R(v)'))
    -- print(parser.stack)
    -- print(parser.links)
    assert(parser.links[1][1] == 0)
    assert(parser.links[2][1] == 4)
    assert(parser.links[3][1] == 4)
    assert(parser.links[4][1] == 1)
    assert(parser.links[5][1] == 4)
end


function test_shortest_stack_oracle()
    local oracle = ShortestStackArcStandardOracle(nil, input_vocabs, unlabeled_actions)
    oracle.parser:reset(tokens)
    
    local action = oracle:oracle()
    assert(action == unlabeled_actions.shift)
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.shift)
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.shift)
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.vocab:get_index('L(v)'))
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.shift)
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.shift)
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.vocab:get_index('R(v)'))
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.vocab:get_index('R(v)'))
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == unlabeled_actions.vocab:get_index('R(v)'))
    oracle.parser:execute(action)
    
    action = oracle:oracle()
    assert(action == nil)
end

function test_build_dataset()
    local oracle = ShortestStackArcStandardOracle(
            ChenManningFeatures(), input_vocabs, unlabeled_actions)
    local x, y = oracle:build_dataset(ds, '', 100)
    assert(#x == 2)
    assert(x[1]:size()[1] == y:size()[1])
    assert(x[2]:size()[1] == y:size()[1])
end

function test_load_stanford_model()
    local path = 'stanford-parser-full-2014-10-31/PTB_Stanford_params.txt.gz'
    mlp, vocab, actions = load_stanford_model(path)
    print(mlp)
    assert(mlp)
    assert(vocab)
    assert(actions)
end

test_load_stanford_model()
test_execute()
test_shortest_stack_oracle()
test_build_dataset()
print("OK")