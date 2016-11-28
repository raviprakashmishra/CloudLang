require('data')
require('util')
require('math')
  
local _ChenManningFeatures = torch.class('ChenManningFeatures')

    function _ChenManningFeatures:__call(x, tokens, links, stack, buffer, vocabs)
        -- notice: a[{1, 2}] is slightly faster than a[1][2]
        -- and a1[2] (with a1 preset to be a[1]) is faster than both
        -- this method must match that of Stanford nndep parser so that
        -- we can load Stanford's model into our parser
        -- see https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/parser/nndep/DependencyParser.java#L206

        x:narrow(1, 1, 18):fill(vocabs.word:get_index('__NONE__'))
        x:narrow(1, 19, 18):fill(vocabs.pos:get_index('__NONE__'))
        x:narrow(1, 37, 12):fill(vocabs.label:get_index('__NONE__'))
        
        if #stack >= 1 then
            x[3] = tokens[{stack[1], 1}]    -- word(s1)
            x[21] = tokens[{stack[1], 2}]   -- pos(s1)

            local left11 = cget_left_dependent(links, stack[1], 1) 
            if left11 then
                x[7] = tokens[{left11, 1}]  -- word(lc1(s1))
                x[25] = tokens[{left11, 2}] -- pos(lc1(s1))
                x[37] = links[{left11, 2}]
                local left_left = cget_left_dependent(links, left11, 1) 
                if left_left then
                    -- print(left_left)
                    x[11] = tokens[{left_left, 1}]  -- word(lc1(lc1(s1)))
                    x[29] = tokens[{left_left, 2}]  -- pos(lc1(lc1(s1)))
                    x[41] = links[{left_left, 2}]
                end
                local left12 = cget_left_dependent(links, stack[1], 2) 
                if left12 then
                    x[9] = tokens[{left12, 1}]  -- word(lc2(s1))
                    x[27] = tokens[{left12, 2}] -- pos(lc2(s1))
                    x[39] = links[{left12, 2}]
                end
            end
                
            local right11 = cget_right_dependent(links, stack[1], 1)
            if right11 then
                x[8] = tokens[{right11, 1}]     -- word(rc1(s1))
                x[26] = tokens[{right11, 2}]    -- pos(rc1(s1))
                x[38] = links[{right11, 2}]
                local right_right = cget_right_dependent(links, right11, 1) 
                if right_right then
                    x[12] = tokens[{right_right, 1}]    -- word(rc1(rc1(s1)))
                    x[30] = tokens[{right_right, 2}]    -- pos(rc1(rc1(s1)))
                    x[42] = links[{right_right, 2}]
                end
                local right12 = cget_right_dependent(links, stack[1], 2) 
                if right12 then
                    x[10] = tokens[{right12, 1}]    -- word(rc2(s1))
                    x[28] = tokens[{right12, 2}]    -- pos(rc2(s1))
                    x[40] = links[{right12, 2}]
                end
            end
            
            if #stack >= 2 then 
                x[2] = tokens[{stack[2], 1}]    -- word(s2)
                x[20] = tokens[{stack[2], 2}]   -- pos(s2)
                
                local left21 = cget_left_dependent(links, stack[2], 1)  
                if left21 then
                    x[13] = tokens[{left21, 1}]   -- word(lc1(s2))
                    x[31] = tokens[{left21, 2}]   -- pos(lc1(s2))
                    x[43] = links[{left21, 2}]
                    local left_left = cget_left_dependent(links, left21, 1) 
                    if left_left then
                        x[17] = tokens[{left_left, 1}] -- word(lc1(lc1(s2)))
                        x[35] = tokens[{left_left, 2}] -- pos(lc1(lc1(s2)))
                        x[47] = links[{left_left, 2}]
                    end
                    local left22 = cget_left_dependent(links, stack[2], 2) 
                    if left22 then
                        x[15] = tokens[{left22, 1}] -- word(lc2(s2))
                        x[33] = tokens[{left22, 2}] -- pos(lc2(s2))
                        x[45] = links[{left22, 2}]
                    end
                end
                
                local right21 = cget_right_dependent(links, stack[2], 1)
                if right21 then
                    x[14] = tokens[{right21, 1}] -- word(rc1(s2))
                    x[32] = tokens[{right21, 2}] -- pos(rc1(s2))
                    x[44] = links[{right21, 2}]

                    local right_right = cget_right_dependent(links, right21, 1) 
                    if right_right then
                        x[18] = tokens[{right_right, 1}] -- word(rc1(rc1(s2)))
                        x[36] = tokens[{right_right, 2}] -- pos(rc1(rc1(s2)))
                        x[48] = links[{right_right, 2}]
                    end
                    local right22 = cget_right_dependent(links, stack[2], 2)
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

    function _ChenManningFeatures:num()
        return 48
    end
    