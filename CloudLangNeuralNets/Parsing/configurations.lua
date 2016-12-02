require('paths')
require('util')
require('torch')
require('math')

--To run sequentially
io.output():setvbuf("no")

local _ChenNManningFeatures = torch.class('ChenNManningFeatures')

    function _ChenNManningFeatures:__call(x, tokens, links, stack, buffer, vocabs)
        x:narrow(1, 1, 18):fill(vocabs.word:get_index('__NONE__'))
        x:narrow(1, 19, 18):fill(vocabs.pos:get_index('__NONE__'))
        x:narrow(1, 37, 12):fill(vocabs.label:get_index('__NONE__'))
        
        if #stack >= 1 then
            x[3] = tokens[{stack[1], 1}]    -- word(s1)
            x[21] = tokens[{stack[1], 2}]   -- pos(s1)

            local left11 = left_dependency(links, stack[1], 1) 
            if left11 then
                x[7] = tokens[{left11, 1}]  -- word(lc1(s1))
                x[25] = tokens[{left11, 2}] -- pos(lc1(s1))
                x[37] = links[{left11, 2}]
                local left_left = left_dependency(links, left11, 1) 
                if left_left then
                    -- print(left_left)
                    x[11] = tokens[{left_left, 1}]  -- word(lc1(lc1(s1)))
                    x[29] = tokens[{left_left, 2}]  -- pos(lc1(lc1(s1)))
                    x[41] = links[{left_left, 2}]
                end
                local left12 = left_dependency(links, stack[1], 2) 
                if left12 then
                    x[9] = tokens[{left12, 1}]  -- word(lc2(s1))
                    x[27] = tokens[{left12, 2}] -- pos(lc2(s1))
                    x[39] = links[{left12, 2}]
                end
            end
                
            local right11 = right_dependency(links, stack[1], 1)
            if right11 then
                x[8] = tokens[{right11, 1}]     -- word(rc1(s1))
                x[26] = tokens[{right11, 2}]    -- pos(rc1(s1))
                x[38] = links[{right11, 2}]
                local right_right = right_dependency(links, right11, 1) 
                if right_right then
                    x[12] = tokens[{right_right, 1}]    -- word(rc1(rc1(s1)))
                    x[30] = tokens[{right_right, 2}]    -- pos(rc1(rc1(s1)))
                    x[42] = links[{right_right, 2}]
                end
                local right12 = right_dependency(links, stack[1], 2) 
                if right12 then
                    x[10] = tokens[{right12, 1}]    -- word(rc2(s1))
                    x[28] = tokens[{right12, 2}]    -- pos(rc2(s1))
                    x[40] = links[{right12, 2}]
                end
            end
            
            if #stack >= 2 then 
                x[2] = tokens[{stack[2], 1}]    -- word(s2)
                x[20] = tokens[{stack[2], 2}]   -- pos(s2)
                
                local left21 = left_dependency(links, stack[2], 1)  
                if left21 then
                    x[13] = tokens[{left21, 1}]   -- word(lc1(s2))
                    x[31] = tokens[{left21, 2}]   -- pos(lc1(s2))
                    x[43] = links[{left21, 2}]
                    local left_left = left_dependency(links, left21, 1) 
                    if left_left then
                        x[17] = tokens[{left_left, 1}] -- word(lc1(lc1(s2)))
                        x[35] = tokens[{left_left, 2}] -- pos(lc1(lc1(s2)))
                        x[47] = links[{left_left, 2}]
                    end
                    local left22 = left_dependency(links, stack[2], 2) 
                    if left22 then
                        x[15] = tokens[{left22, 1}] -- word(lc2(s2))
                        x[33] = tokens[{left22, 2}] -- pos(lc2(s2))
                        x[45] = links[{left22, 2}]
                    end
                end
                
                local right21 = right_dependency(links, stack[2], 1)
                if right21 then
                    x[14] = tokens[{right21, 1}] -- word(rc1(s2))
                    x[32] = tokens[{right21, 2}] -- pos(rc1(s2))
                    x[44] = links[{right21, 2}]

                    local right_right = right_dependency(links, right21, 1) 
                    if right_right then
                        x[18] = tokens[{right_right, 1}] -- word(rc1(rc1(s2)))
                        x[36] = tokens[{right_right, 2}] -- pos(rc1(rc1(s2)))
                        x[48] = links[{right_right, 2}]
                    end
                    local right22 = right_dependency(links, stack[2], 2)
                    if right22 then
                        x[16] = tokens[{right22, 1}] -- word(rc2(s2))
                        x[34] = tokens[{right22, 2}] -- pos(rc2(s2))
                        x[46] = links[{right22, 2}]
                    end
                end

                if #stack >= 3 then 
                    x[1] = tokens[{stack[3], 1}]    -- word(s3)
                    x[19] = tokens[{stack[3], 2}]   -- pos(s3)
                end
            end
        end       
        for j = 1, math.min(3, #buffer) do
            x[3 + j] = tokens[{buffer[j], 1}]  -- word(b_j)
            x[21 + j] = tokens[{buffer[j], 2}] -- pos(b_j)
        end
    end

    function _ChenNManningFeatures:num()
        return 48
    end

home_dir = paths.dirname(paths.thisfile())
data_dir = paths.concat(home_dir, 'treebank')
out_dir = 'output/dep'
create_directories(out_dir)

stanford_deps_loc = paths.concat(out_dir, 'treebank.dep')
stanford_vocab_loc = paths.concat(out_dir, 'vocab.th7')
stanford_actions_loc = paths.concat(out_dir, 'actions.th7')
train_data_loc = paths.concat(out_dir, 'train_model.th7')
valid_data_loc = paths.concat(out_dir, 'valid.th7')
test_data_loc = paths.concat(out_dir, 'test.th7')

root_dep_label = 'root'

rand_seed = 20141114
if rand_seed then 
    torch.manualSeed(rand_seed)
end

chen_manning_features = {
    chen_manning = ChenNManningFeatures(),
}
