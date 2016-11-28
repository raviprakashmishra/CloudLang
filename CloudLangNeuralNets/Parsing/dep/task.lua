require('torch')
require('util')
require('data')
require('paths')
require('model')
require('dep.config')

local function check_right_dependency(gold_links, pred_links, s1)
    if cget_right_dependent(gold_links, s1) == nil then
        return true;
    elseif cget_right_dependent(pred_links, s1) ~= nil then
        if cget_right_dependent(gold_links, s1) == cget_right_dependent(pred_links, s1) then
            return true;
        end
    end
    return false;
end

local _Parser = torch.class('Parser')

    function _Parser:reset()
        error('Not implemented')        
    end

    function _Parser:predict() 
    end
    
    function _Parser:execute(action) 
        error('Not implemented')        
    end


local function indices2mask(indices, mask)
    -- receive an array of integers or set of integers
    -- set elements corresponding to those integers to 1, leave others untouched 
    for _, v in ipairs(indices) do
        if type(v) == 'number' then
            mask[v] = 1
        elseif type(v) == 'table' then
            for i, _ in pairs(v) do
                mask[i] = 1
            end
        else
            error('Unsupported value: ' .. v)
        end
    end
end

local _ArcStandardParser, parent = torch.class('ArcStandardParser', 'Parser')

    function _ArcStandardParser:__init(actions)
        assert(actions.left and actions.right and actions.right_root and actions.shift)
        self.actions = actions
        self.masks = torch.ByteTensor(5, max_index(actions.vocab))
        self.masks:zero()
        self.states = {
            terminal = 0,
            left_right_shift = 1,
            right_root_shift = 2,
            right_root = 3,
            left_right = 4,
            shift = 5,
        }
        indices2mask({actions.left, actions.right, actions.shift}, self.masks[self.states.left_right_shift])
        indices2mask({actions.shift}, self.masks[self.states.shift])
        indices2mask({actions.left, actions.right}, self.masks[self.states.left_right])
        indices2mask({actions.right_root, actions.shift}, self.masks[self.states.right_root_shift])
        indices2mask({actions.right_root}, self.masks[self.states.right_root])
    end

    function _ArcStandardParser:reset(tokens)
        self.tokens = tokens
        self.stack = {}
        self.buffer = list(range(tokens:size(1)))
        self.links = torch.LongTensor(tokens:size(1), 5)
        self.links:zero()
    end
    
    function _ArcStandardParser:state()
        if #self.buffer >= 1 then
            if #self.stack > 2 then
                return self.states.left_right_shift
            else
                return self.states.shift
            end
        else
            if #self.stack > 2 then
                return self.states.left_right
            elseif #self.stack == 2 then
                assert(self.stack[1] ~= 1 and self.stack[2] == 1,
                        "Expecting root followed by something else in the stack")
                return self.states.right_root
            end
        end
        return self.states.terminal
    end
    
    function _ArcStandardParser:execute(a)
        s1, s2 = self.stack[1], self.stack[2]
        if self.actions.left[a] then
            assert(#self.stack >= 2)
            self.links[{s2, 1}] = s1
            self.links[{s2, 2}] = self.actions.action2up[a] or self.actions.action2rel[a]
            self.links[{s2, 3}] = self.actions.action2down[a] or self.actions.action2rel[a]
            -- 4th column is leftmost child, 5th is siblings
            self.links[{s1, 4}] = ll_insert(self.links:select(2, 5), self.links[{s1, 4}], s2)
            table.remove(self.stack, 2)
        elseif self.actions.right[a] or a == self.actions.right_root then
            assert(#self.stack >= 2)
            self.links[{s1, 1}] = s2
            self.links[{s1, 2}] = self.actions.action2up[a] or self.actions.action2rel[a]
            self.links[{s1, 3}] = self.actions.action2down[a] or self.actions.action2rel[a]
            -- 4th column is leftmost child, 5th is siblings
            self.links[{s2, 4}] = ll_insert(self.links:select(2, 5), self.links[{s2, 4}], s1)
            table.remove(self.stack, 1)
        elseif a == self.actions.shift then
            assert(#self.buffer >= 1)
            table.insert(self.stack, 1, table.remove(self.buffer, 1))
        else
            error('Unsupported action: ' .. tostring(a))
        end
    end

local _ShortestStackArcStandardOracle = torch.class('ShortestStackArcStandardOracle')

    function ShortestStackArcStandardOracle:__init(feature_extractor, input_vocabs, actions)
        self.feature_extractor = feature_extractor
        self.input_vocabs = input_vocabs
        self.parser = ArcStandardParser(actions)
    end

    function ShortestStackArcStandardOracle:oracle()
        --[[
            based on Malt parser's implementation:
            http://grepcode.com/file/repo1.maven.org/maven2/org.maltparser/maltparser/1.8/org/maltparser/parser/algorithm/nivre/ArcStandardOracle.java#ArcStandardOracle
        --]]
        local debug = false
        local tokens = self.parser.tokens
        local actions = self.parser.actions
        local buffer = self.parser.buffer
        local stack = self.parser.stack
        local gold_links = tokens:narrow(2, 3, 3)
        if #stack < 2 then
            if #buffer <= 0 then
                assert(stack[1] == 1)
                return nil
            else
                if debug then print('shift') end
                return actions.shift
            end
        else
            s1, s2 = stack[1], stack[2]
            if tokens[{s2, 3}] == s1 then
                if debug then print('left') end
                return actions.rel2left[tokens[{s2, 4}]]
            elseif tokens[{s1, 3}] == s2 and 
                    check_right_dependency(gold_links, self.parser.links, s1) then
                if debug then print('right') end
                return actions.rel2right[tokens[{s1, 4}]]
            elseif #buffer >= 1 then
                if debug then print('shift') end
                return actions.shift
            else
                error("You shouldn't see this! Probably unprojective tree was used.")
            end
        end
    end

    function _ShortestStackArcStandardOracle:build_dataset(ds, name, max_rows) 
        name = name or '[noname]'
        print(string.format("Building dataset '%s'... ", name))
        start = os.time()
        local x = torch.LongTensor(max_rows, self.feature_extractor:num())
        local states = torch.LongTensor(max_rows)
        local y = torch.LongTensor(max_rows)
        x:fill(2) -- __NONE__
        local count = 0
        for s = 1, ds.sents:size(1) do
            local tokens = ds.tokens:narrow(1, ds.sents[{s, 2}], ds.sents[{s, 3}])
            if is_projective(tokens:select(2, 3)) then
                assert(tokens[{1, 3}] == 0) -- first token must be root
                self.parser:reset(tokens)
                self.parser:execute(self.parser.actions.shift) -- skip first 2 trivial actions
                self.parser:execute(self.parser.actions.shift)
                -- print(table.concat(self.parser.stack, ','), '\t', table.concat(self.parser.buffer, ','))
                local action = self:oracle()
                while action ~= nil do
                    count = count + 1
                    self.feature_extractor(x[count], tokens, self.parser.links, 
                            self.parser.stack, self.parser.buffer, self.input_vocabs)
                    states[count] = self.parser:state()
                    y[count] = action
                    assert(self.parser.masks[states[count]][y[count]])
                    self.parser:execute(action)
                    -- print(table.concat(self.parser.stack, ','), '\t', table.concat(self.parser.buffer, ','))
                    action = self:oracle()
                    if count % 100000 == 0 then
                        print(count .. ' examples...')
                    end
                end
            end
        end
        x = x:narrow(1, 1, count)
        states = states:narrow(1, 1, count)
        y = y:narrow(1, 1, count)
        collectgarbage() -- important! avoid memory error
        stop = os.time()
        print(string.format("Building dataset '%s'... Done (%d examples, %d s).", 
                name, y:size(1), stop-start))
        return {x, states}, y
    end
    

local _BatchGreedyArcStandardParser = torch.class('BatchGreedyArcStandardParser')

    function _BatchGreedyArcStandardParser:__init(input_vocabs, actions, 
            feature_extractor, mlp, cuda, report_frequency)
        self.input_vocabs = input_vocabs
        self.actions = actions
        self.feature_extractor = feature_extractor
        self.mlp = mlp
        self.cuda = cuda
        self.report_frequency = report_frequency or 1000
        self.masks = ArcStandardParser(actions).masks
    end

    function _BatchGreedyArcStandardParser:predict(ds, batch_size)
        local s = 1
        local x = torch.LongTensor(batch_size, self.feature_extractor:num())
        local states = torch.LongTensor(batch_size):fill(1)
        local output = torch.LongTensor(ds.tokens:size(1), 3):zero()
        local scores = torch.Tensor() -- to be set by mlp
        local batch_masks = torch.ByteTensor()
        local start_time = os.time()

        local single_parse = function(index)
            local parser = ArcStandardParser(self.actions)
            while s <= ds.sents:size(1) do
                local local_s = s
                s = s + 1
                local start = ds.sents[{local_s, 2}]
                local size = ds.sents[{local_s, 3}]
                local tokens = ds.tokens:narrow(1, start, size) 
                parser:reset(tokens)
                -- skip just 1 first trivial actions instead of 2 to match
                -- the procedure of Stanford neural parser
                parser:execute(self.actions.shift) 
                -- parser:execute(self.actions.shift)
                while parser:state() ~= parser.states.terminal do
                    states[index] = parser:state()
                    self.feature_extractor(x[index], tokens, parser.links, 
                            parser.stack, parser.buffer, self.input_vocabs)
                    coroutine.yield() -- wait for result
                    local max_score, y = scores[index]:max(1)
                    -- io.write(string.format('*** %.1f %s ***\n', max_score[1], parser.actions.vocab:get_word(y[1])))
                    parser:execute(y[1])
                end
                -- io.write('--------------------------------------\n')
                if not (#parser.stack == 1 and #parser.buffer == 0 and parser.stack[1] == 1) then
                    -- error is not handled inside corountine
                    print('WARN: Unfinished parse.')
                end
                output:narrow(1, start, size):copy(parser.links:narrow(2, 1, 3))
                
                if self.report_frequency > 0 and local_s % self.report_frequency == 0 then
                    print(string.format("Sentence %d, speed = %.2f sentences/s", local_s, local_s/(os.time()-start_time)))
                end
            end
        end
        
        local workers = {}
        for i = 1, batch_size do
            workers[i] = coroutine.create(function() single_parse(i) end) 
        end
        local running = true
        while running do
            x:fill(2) -- fill with __NONE__
            running = false
            for i, w in ipairs(workers) do
                if coroutine.status(w) == 'suspended' then
                    assert(coroutine.resume(w))
                end
                running = running or (coroutine.status(w) == 'suspended') 
            end
            if running then
                local probs
                if self.cuda then 
                    probs = self.mlp:forward(x:cuda()):float()
                else
                    probs = self.mlp:forward(x)
                end
                batch_masks:index(self.masks, 1, states)
                scores:resizeAs(probs):fill(-math.huge)
                scores[batch_masks] = probs[batch_masks]
            end
        end
        return output
    end
    
function measure_uas(ds, parser, batch_size, punc)
    batch_size = batch_size or 10000
    local output = parser:predict(ds, batch_size)
    local gold_heads = ds.tokens:select(2, 3)
    local pred_heads = output:select(2, 1)
    if not punc then
        -- match Stanford implementation here:
        -- https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/parser/nndep/ParsingSystem.java#L130
        local mask = torch.ByteTensor(gold_heads:size(1)):fill(1)
        for _, p in ipairs{"``", "''", ".", ",", ":"} do
            local index = parser.input_vocabs.pos:get_index(p)
            mask:cmul(ds.tokens:select(2, 2):ne(index))
        end
        gold_heads = gold_heads[mask]
        pred_heads = pred_heads[mask]
    end
    local correct = pred_heads:eq(gold_heads):sum()
    return correct / gold_heads:size(1)
end


function custom_adagrad(opfunc, x, config, state)
   -- (0) get/update state
   if config == nil and state == nil then
      print('no state table, ADAGRAD initializing')
   end
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (3) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
      
   -- (4) parameter update with single or individual learning rates
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end
   state.paramVariance:addcmul(1,dfdx,dfdx)
   state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):add(1e-6):sqrt()
   x:addcdiv(-clr, dfdx,state.paramStd)

   -- (5) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

function train_parser(cfg, parser, train_x, train_y, valid_x, valid_y, valid_ds, initRange)
    print(string.format("Training %s...", cfg.model_name))
    local start = os.time()
    local example_count = 0
    local batch_count = math.max(train_y:size(1)/cfg.training_batch_size, 1)
    local best_uas = nil
    local params, grad_params = cfg.mlp:getParameters()
    local params2 = torch.Tensor():typeAs(params) -- avoid allocating multiple times
    local train_sx, train_sy -- shuffled x and y (avoid allocating multiple times)
    local feval = function()
        cfg.mlp:training() -- affects dropout layer, if present
        local batch_x, batch_y = random_batch(train_sx, train_sy, cfg.training_batch_size)
        local cost = cfg.criterion:forward(cfg.mlp:forward(batch_x), batch_y)
        assert(tostring(cost) ~= 'nan' and not tostring(cost):find('inf'))
        grad_params:zero()
        cfg.mlp:backward(batch_x, cfg.criterion:backward(cfg.mlp.output, batch_y))
        if cfg.l1_weight and cfg.l1_weight > 0 then
            for _, h in ipairs(cfg.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * cfg.l1_weight 
                h.gradWeight:add(cfg.l1_weight, torch.sign(h.weight))
            end
        end
        if cfg.l2_weight and cfg.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * cfg.l2_weight / 2 
            grad_params:add(cfg.l2_weight, params)
        end
        if cfg.max_grad and cfg.max_grad > 0 then
            grad_params:clamp(-cfg.max_grad, cfg.max_grad)
        end
        return cost, grad_params 
    end
    local monitoring_cost = function(x, y)
        cfg.mlp:evaluate() -- affects dropout layer, if present
        local cost = 0
        local batch_count = math.max(y:size(1)/cfg.monitoring_batch_size, 1)
        for i = 1, batch_count do
            local batch_start = (i-1)*cfg.monitoring_batch_size + 1
            local actual_batch_size = math.min(cfg.monitoring_batch_size, 
                    y:size(1) - batch_start + 1)
            local batch_x = narrow_all(x, 1, batch_start, actual_batch_size)
            local batch_y = y:narrow(1, batch_start, actual_batch_size)
            local c = cfg.criterion:forward(cfg.mlp:forward(batch_x), batch_y)
            cost = cost + c * actual_batch_size
        end
        cost = cost / y:size(1)
        if cfg.l1_weight and cfg.l1_weight > 0 then
            for _, h in ipairs(cfg.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * cfg.l1_weight 
            end
        end
        if cfg.l2_weight and cfg.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * cfg.l2_weight / 2 
        end
        return cost
    end 
    local optim_state = {
            learningRate = cfg.learningRate, 
            learningRateDecay = cfg.learningRateDecay,
            weightDecay = 0, 
            momentum = 0,
    }
    assert(optim_state.learningRate > 0)
    assert(optim_state.learningRateDecay >= 0)
    if initRange then
        params:uniform(-initRange, initRange)
    end
    for epoch_count = 1, cfg.max_epochs do
        train_sx, train_sy = shuffle2(train_x, train_y, train_sx, train_sy)
        for batch_no = 1, batch_count do
            custom_adagrad(feval, params, optim_state)
            
            example_count = example_count + cfg.training_batch_size
            if cfg.batch_reporting_frequency > 0 and
                    batch_no % cfg.batch_reporting_frequency == 0 then
                local speed = example_count / (os.time()-start)
                io.write(string.format("Batch %d, speed = %.2f examples/s\r", batch_no, speed))
            end
        end
        if cfg.batch_reporting_frequency > 0 and
                batch_count >= cfg.batch_reporting_frequency then
            io.write('\n')
        end
        collectgarbage() -- important! avoid memory error

        if cfg.epoch_reporting_frequency > 0 and 
                epoch_count % cfg.epoch_reporting_frequency == 0 then
            print(string.format("Epoch %d:", epoch_count))
            print(string.format("\tTraining cost: %f", 
                    monitoring_cost(train_sx, train_sy)))
            if valid_x ~= nil and valid_y ~= nil then
                print(string.format("\tValid cost: %f", 
                        monitoring_cost(valid_x, valid_y)))
            end
            print(string.format("\tSeconds since start: %d s", os.time()-start))
        end
        
        if valid_ds and cfg.best_path then
            cfg.mlp:evaluate() -- affects dropout layer, if present
            local uas = measure_uas(valid_ds, parser, cfg.monitoring_batch_size, cfg.punc)
            io.write(string.format('UAS on development set: %.7f\n', uas))
            if best_uas == nil or uas > best_uas then
                io.write(string.format('Writing best model to %s... ', cfg.best_path)) 
                torch.save(cfg.best_path, cfg.mlp)
                best_uas = uas
                print('Done.')
            end
        end
    end
    
    if valid_x ~= nil and valid_y ~= nil and cfg.best_path ~= nil then
        io.write(string.format('Loading from %s... ', cfg.best_path))
        cfg.best_mlp = torch.load(cfg.best_path)
        print('Done.')
    end
    local stop = os.time()
    print(string.format("Training %s... Done (%.2f min).", cfg.model_name, (stop-start)/60.0))
end

local function pad_with_slash(s)
    if not s:endsWith('/') then
        return s .. '/'
    else
        return s
    end
end
    
function penn2dep_lth(inp_dir_or_file, out_dir_or_file)
    print("in penn2dep_lth")
    print(inp_dir_or_file)
    print(out_dir_or_file)
    if paths.filep(inp_dir_or_file) then
        mkdirs(paths.dirname(out_dir_or_file))
        local cmd = string.format('java -jar pennconverter/pennconverter.jar ' ..
                '-stopOnError=False -raw < %s > %s.dep', inp_dir_or_file, out_dir_or_file)
        print(cmd)
        assert(os.execute(cmd))
    else
        inp_dir_or_file = pad_with_slash(inp_dir_or_file)
        out_dir_or_file = pad_with_slash(out_dir_or_file) 
        for inp_path in find_files_recursively(inp_dir_or_file) do
            local out_path = inp_path:gsub(inp_dir_or_file:quote(), out_dir_or_file)
            penn2dep_lth(inp_path, out_path)
        end
    end
end
    
function penn2sd(inp_dir_or_file, out_dir_or_file)
    print("in penn2sd")
    print(inp_dir_or_file)
    print(out_dir_or_file)
    if paths.filep(inp_dir_or_file) then
        mkdirs(paths.dirname(out_dir_or_file))
        local cmd = 'java -cp stanford-parser-full-2014-10-31/stanford-parser.jar ' ..
                'edu.stanford.nlp.trees.EnglishGrammaticalStructure -maxLength 100 -basic -conllx ' ..
                '-treeFile ' .. inp_dir_or_file .. ' > ' .. out_dir_or_file .. '.dep'
        print(cmd) 
        assert(os.execute(cmd))
    else
        inp_dir_or_file = pad_with_slash(inp_dir_or_file)
        out_dir_or_file = pad_with_slash(out_dir_or_file) 
        for inp_path in find_files_recursively(inp_dir_or_file) do
            local out_path = inp_path:gsub(inp_dir_or_file:quote(), out_dir_or_file)
            penn2sd(inp_path, out_path)
        end
    end
end

function jackknife(path, jkf_path, err_path, model_path, append)
    local changed = 0
    local total = 0
    mkdirs(paths.dirname(jkf_path))
    local tmp = os.tmpname()
    local spt = 'stanford-postagger-2015-12-09'
    local class_path = spt .. '/stanford-postagger-3.6.0.jar:' .. spt .. '/lib/slf4j-simple.jar:' .. spt .. '/lib/slf4j-api.jar'
    local cmd = 'java -classpath ' .. class_path .. ' edu.stanford.nlp.tagger.maxent.MaxentTagger ' ..
            '-props stanford-postagger-2015-12-09/penn-treebank.props ' ..
            '-model ' .. model_path .. ' -testFile format=TREES,' .. path ..
            ' > ' .. tmp .. ' 2> ' .. err_path 
    -- print('Command: ' .. cmd)
    assert(os.execute(cmd))
    local f = io.open(jkf_path, ternary(append, 'a', 'w'))
    for tree_line, tmp_line in iter_zip(penn_sentences(path), io.lines(tmp)) do
        local word_pos_iter = list_iter(split(tmp_line, ' ')) 
        tree_line = tree_line:gsub('%((%S+) (%S+)%)', 
                function(tree_pos, tree_word) 
                    if tree_pos == '-NONE-' then
                        -- keep original POS tag
                        return string.format('(%s %s)', tree_pos, tree_word)
                    else
                        -- replace by predicted POS tag
                        local tmp_word, tmp_pos = table.unpack(split(word_pos_iter(), '_'))
                        if tree_word:gsub('\\/', '/'):gsub('\\%*', '*') ~= tmp_word and 
                                tree_word ~= 'theatre' and tree_word ~= 'august' and
                                tree_word ~= 'neighbours' then
                            print(string.format('WARN: Differing original and ' .. 
                                    'POS tagged words: %s vs. %s', tree_word, tmp_word))
                        end
                        if tree_pos ~= tmp_pos then 
                            changed = changed + 1 
                        end
                        total = total + 1
                        return string.format('(%s %s)', tmp_pos, tree_word)
                    end
                end)
        assert(word_pos_iter() == nil)
        f:write(tree_line):write('\n')
    end
    f:close()
    print(string.format('Changed: %d, total: %d, wrote to: %s', changed, total, jkf_path))
    os.execute('grep "Total tags right:" ' .. err_path)
    os.remove(tmp)
end

function make_actions(rels, label_vocab, directed_labels)
    local actions = {
        vocab = Vocabulary(),
        left = {},          -- set of all left actions (except root)
        right = {},         -- set of all right actions (except root) 
        rel2left = {},      -- map from relation (label) to left actions (except root) 
        rel2right = {},     -- map from relation (label) to right actions (except root) 
        action2rel = {},    -- map from actions (including root) to undirected relation (label)
        action2up = {},     -- map from actions (including root) to upward relation (label)
        action2down = {},   -- map from actions (including root) to downward relation (label)
    }
    -- keep the same order as makeTransitions in Stanford's parser 
    for _, rel in ipairs(rels) do
        if rel ~= '__NONE__' then
            actions.vocab:get_index('L(' .. rel .. ')')
        end
    end
    for _, rel in ipairs(rels) do
        if rel ~= '__NONE__' then
            actions.vocab:get_index('R(' .. rel .. ')')
        end
    end
    actions.vocab:get_index('S')
    -- build some structures
    actions.right_root = actions.vocab:get_index('R(' .. root_dep_label .. ')')
    for _, rel in ipairs(rels) do
        if rel ~= '__NONE__' then
            local l = actions.vocab:get_index('L(' .. rel .. ')')
            local r = actions.vocab:get_index('R(' .. rel .. ')')
            if directed_labels then
                local u = label_vocab:get_index(rel .. '#U')
                local d = label_vocab:get_index(rel .. '#D')
                if u ~= 1 then
                    if rel ~= root_dep_label then
                        actions.rel2left[u] = l 
                        actions.rel2right[u] = r
                    end
                    actions.action2up[l] = u
                    actions.action2up[r] = u
                end
                if d ~= 1 then 
                    if rel ~= root_dep_label then
                        actions.rel2left[d] = l 
                        actions.rel2right[d] = r
                    end
                    actions.action2down[l] = d
                    actions.action2down[r] = d
                end
            else
                local rel = label_vocab:get_index(rel)
                if rel ~= root_dep_label then
                    actions.rel2left[rel] = l 
                    actions.rel2right[rel] = r
                end
                actions.action2rel[l] = rel
                actions.action2rel[r] = rel
            end
            if rel ~= root_dep_label then
                actions.left[l] = true
                actions.right[r] = true
            end
        end
    end
    actions.shift = actions.vocab:get_index('S')
    return actions
end

function translate_magic_words(s)
    s = s:gsub('%-NULL%-', '__NONE__'):gsub('%-UNKNOWN%-', '__MISSING__'):gsub('%-ROOT%-', '__ROOT__')
    return s
end

function load_stanford_model(path)
    if path:find('%.gz$') then
        local unzipped = paths.tmpname()
        assert(os.execute(string.format('gunzip %s -c > %s', path, unzipped)))
        path = unzipped
    end
    local start = os.time()
    io.write("Loading depparse model file: " .. path .. "... ")
    local f = io.open(path) 
    local s = f:read('*line')
    local nDict = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local nPOS = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local nLabel = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local eSize = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local hSize = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local nTokens = tonumber(s:sub(s:find('=') + 1))
    s = f:read('*line')
    local nPreComputed = tonumber(s:sub(s:find('=') + 1))
    local mlp = new_cubenet(nDict+nPOS+nLabel, eSize, nTokens, hSize, nLabel * 2 - 1)

    local indexer = Indexer()
    local vocabs = {
        word = Vocabulary(indexer),
        pos = Vocabulary(indexer),
        label = Vocabulary(indexer),
    }
    local knownLabels = {}
    local E = mlp:get_lookup_table().weight
    local splits

    for k = 1, nDict do
        local s = f:read('*line')
        local splits = split(s, " ");
        local index = vocabs.word:get_index(translate_magic_words(splits[1]))
        assert(#splits-1 == eSize)
        for i = 1, eSize do
            E[index][i] = tonumber(splits[i + 1]);
        end
    end
    for k = 1, nPOS do
        local s = f:read('*line')
        local splits = split(s, " ");
        local index = vocabs.pos:get_index(translate_magic_words(splits[1]))
        assert(#splits-1 == eSize)
        for i = 1, eSize do
            E[index][i] = tonumber(splits[i + 1]);
        end
    end
    for k = 1, nLabel do
        local s = f:read('*line')
        local splits = split(s, " ");
        local label = translate_magic_words(splits[1])
        local index = vocabs.label:get_index(label);
        table.insert(knownLabels, label)
        assert(#splits-1 == eSize)
        for i = 1, eSize do
            E[index][i] = tonumber(splits[i + 1]);
        end
    end
    assert(max_index(vocabs) == E:size(1))
    
    local W1 = mlp:get_hidden_layers()[1].weight
    for j = 1, W1:size(2) do
        local s = f:read('*line')
        local splits = split(s, " ");
        assert(#splits == W1:size(1))
        for i = 1, W1:size(1) do
            W1[i][j] = tonumber(splits[i]);
        end
    end

    local b1 = mlp:get_hidden_layers()[1].bias
    local s = f:read('*line')
    local splits = split(s, " ");
    assert(#splits == b1:size(1))
    for i = 1, b1:size(1) do
        b1[i] = tonumber(splits[i]);
    end

    local W2 = mlp:get_output_layer().weight
    for j = 1, W2:size(2) do
        local s = f:read('*line')
        local splits = split(s, " ");
        assert(#splits == W2:size(1))
        for i = 1, W2:size(1) do
            W2[i][j] = tonumber(splits[i]);
        end
    end
    -- skip pre-computed weights
    f:close()

    -- equivalent to makeTransitions() 
    local actions = make_actions(knownLabels, vocabs.label)
    -- print(vocabs.label)
    for _, vocab in pairs(vocabs) do
        vocab:seal(true)
    end
    
    print(string.format('Done (%s s).', os.time()-start))
    return mlp, vocabs, actions
end
