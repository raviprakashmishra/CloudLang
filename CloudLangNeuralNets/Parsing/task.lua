require('torch')
require('util')
require('paths')
require('neural_net_model')
require('configurations')

local function check_right_dependency(gold_links, pred_links, s1)
    if right_dependency(gold_links, s1) == nil then
        return true;
    elseif right_dependency(pred_links, s1) ~= nil then
        if right_dependency(gold_links, s1) == right_dependency(pred_links, s1) then
            return true;
        end
    end
    return false;
end

local function mask_indices(indices, mask)
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

local _TransitionParser = torch.class('TransitionParser')

    function _TransitionParser:__init(actions)
        self.actions = actions
        self.masks = torch.ByteTensor(5, get_max_vocab_index(actions.vocab))
        self.masks:zero()
        self.states = {
            terminal = 0,
            left_right_shift = 1,
            right_root_shift = 2,
            right_root = 3,
            left_right = 4,
            shift = 5,
        }
        mask_indices({actions.left, actions.right, actions.shift}, self.masks[self.states.left_right_shift])
        mask_indices({actions.shift}, self.masks[self.states.shift])
        mask_indices({actions.left, actions.right}, self.masks[self.states.left_right])
        mask_indices({actions.right_root, actions.shift}, self.masks[self.states.right_root_shift])
        mask_indices({actions.right_root}, self.masks[self.states.right_root])
    end

    function _TransitionParser:reset_parser(tokens)
        self.tokens = tokens
        self.buffer = list(find_range(tokens:size(1)))
        self.stack = {}
        self.links = torch.LongTensor(tokens:size(1), 5)
        self.links:zero()
    end
    
    function _TransitionParser:current_state()
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
                return self.states.right_root
            end
        end
        return self.states.terminal
    end
    
    function _TransitionParser:action_execute(a)
        s1, s2 = self.stack[1], self.stack[2]
        if self.actions.left[a] then
            self.links[{s2, 1}] = s1
            self.links[{s2, 2}] = self.actions.action2up[a] or self.actions.action2rel[a]
            self.links[{s2, 3}] = self.actions.action2down[a] or self.actions.action2rel[a]
            self.links[{s1, 4}] = insert_linked_list(self.links:select(2, 5), self.links[{s1, 4}], s2)
            table.remove(self.stack, 2)
        elseif self.actions.right[a] or a == self.actions.right_root then
            self.links[{s1, 1}] = s2
            self.links[{s1, 2}] = self.actions.action2up[a] or self.actions.action2rel[a]
            self.links[{s1, 3}] = self.actions.action2down[a] or self.actions.action2rel[a]
            self.links[{s2, 4}] = insert_linked_list(self.links:select(2, 5), self.links[{s2, 4}], s1)
            table.remove(self.stack, 1)
        elseif a == self.actions.shift then
            table.insert(self.stack, 1, table.remove(self.buffer, 1))
        else
            error('Incorrect action ' .. tostring(a))
        end
    end

local _TheOracle = torch.class('TheOracle')

    function TheOracle:__init(feature_extractor, input_vocabs, actions)
        self.feature_extractor = feature_extractor
        self.input_vocabs = input_vocabs
        self.parser = TransitionParser(actions)
    end

    function _TheOracle:create_data_set(ds, name, max_rows) 
        name = name or '[noname]'
        print(string.format("Creating dataset '%s'... ", name))
        start = os.time()
        local x = torch.LongTensor(max_rows, self.feature_extractor:num())
        local states = torch.LongTensor(max_rows)
        local y = torch.LongTensor(max_rows)
        x:fill(2) -- __NONE__
        local count = 0
        for s = 1, ds.sents:size(1) do
            local tokens = ds.tokens:narrow(1, ds.sents[{s, 2}], ds.sents[{s, 3}])
            if is_tree_projective(tokens:select(2, 3)) then
                self.parser:reset_parser(tokens)
                self.parser:action_execute(self.parser.actions.shift)
                self.parser:action_execute(self.parser.actions.shift)
                local action = self:oracle()
                while action ~= nil do
                    count = count + 1
                    self.feature_extractor(x[count], tokens, self.parser.links, 
                            self.parser.stack, self.parser.buffer, self.input_vocabs)
                    states[count] = self.parser:current_state()
                    y[count] = action
                    self.parser:action_execute(action)
                    action = self:oracle()
                    if count % 100000 == 0 then
                        print('Built ' .. count .. ' examples...')
                    end
                end
            end
        end
        x = x:narrow(1, 1, count)
        y = y:narrow(1, 1, count)
        states = states:narrow(1, 1, count)
        collectgarbage()
        stop = os.time()
        print(string.format("Creating dataset '%s'... Done (%d examples, %d s).", 
                name, y:size(1), stop-start))
        return {x, states}, y
    end

    function TheOracle:oracle()
        local tokens = self.parser.tokens
        local actions = self.parser.actions
        local buffer = self.parser.buffer
        local stack = self.parser.stack
        local gold_links = tokens:narrow(2, 3, 3)
        local debug = false
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
                error("Problem with the tree")
            end
        end
    end
    

local _GreedyParser = torch.class('GreedyParser')

    function _GreedyParser:__init(input_vocabs, actions, 
            feature_extractor, mlp, cuda, time_to_print)
        self.input_vocabs = input_vocabs
        self.actions = actions
        self.feature_extractor = feature_extractor
        self.mlp = mlp
        self.cuda = cuda
        self.time_to_print = time_to_print or 1000
        self.masks = TransitionParser(actions).masks
    end

    function _GreedyParser:predict_parse_tree(ds, batch_size)
        local s = 1
        local x = torch.LongTensor(batch_size, self.feature_extractor:num())
        local states = torch.LongTensor(batch_size):fill(1)
        local output = torch.LongTensor(ds.tokens:size(1), 3):zero()
        local batch_masks = torch.ByteTensor()
        local parse_score = torch.Tensor()
        local start_time = os.time()

        local single_parse = function(index)
            local parser = TransitionParser(self.actions)
            while s <= ds.sents:size(1) do
                local local_s = s
                s = s + 1
                local start = ds.sents[{local_s, 2}]
                local size = ds.sents[{local_s, 3}]
                local tokens = ds.tokens:narrow(1, start, size) 
                parser:reset_parser(tokens)
                parser:action_execute(self.actions.shift) 
                while parser:current_state() ~= parser.states.terminal do
                    states[index] = parser:current_state()
                    self.feature_extractor(x[index], tokens, parser.links, 
                            parser.stack, parser.buffer, self.input_vocabs)
                    coroutine.yield()
                    local max_score, y = parse_score[index]:max(1)
                    parser:action_execute(y[1])
                end
                if not (#parser.stack == 1 and #parser.buffer == 0 and parser.stack[1] == 1) then
                    print('Parsing didn\'t finish')
                end
                output:narrow(1, start, size):copy(parser.links:narrow(2, 1, 3))
                
                if self.time_to_print > 0 and local_s % self.time_to_print == 0 then
                    print(string.format("Parsed %d sentences at a speed of %.2f sentences/s", local_s, local_s/(os.time()-start_time)))
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
                parse_score:resizeAs(probs):fill(-math.huge)
                parse_score[batch_masks] = probs[batch_masks]
            end
        end
        return output
    end

function adagrad_mod(opfunc, x, config, state)
   local config = config or {}
   local state = state or config
   local learningRateDecay = config.learningRateDecay or 0
   local learningRate = config.learningRate or 1e-3
   state.evaluationCount = state.evaluationCount or 0
   local nevals = state.evaluationCount
   local fx,dfdx = opfunc(x)
   local cumulative_learningRate = learningRate / (1 + nevals*learningRateDecay)
      
   if not state.paramVariance then
      state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
      state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
   end
   state.paramVariance:addcmul(1,dfdx,dfdx)
   state.paramStd:resizeAs(state.paramVariance):copy(state.paramVariance):add(1e-6):sqrt()
   x:addcdiv(-cumulative_learningRate, dfdx,state.paramStd)

   state.evaluationCount = state.evaluationCount + 1

   return x,{fx}
end
    
function compute_UAS(ds, parser, batch_size, punc)
    batch_size = batch_size or 10000
    local output = parser:predict_parse_tree(ds, batch_size)
    local gold_heads = ds.tokens:select(2, 3)
    local pred_heads = output:select(2, 1)
    if not punc then
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

function train_neural_parser(neural_net_config, parser, train_x, train_y, valid_x, valid_y, valid_ds, initRange)
    print(string.format("Training %s...", neural_net_config.model_name))
    local start = os.time()
    local example_count = 0
    local batch_count = math.max(train_y:size(1)/neural_net_config.training_batch_size, 1)
    local best_uas = nil
    local params, grad_params = neural_net_config.mlp:getParameters()
    local params2 = torch.Tensor():typeAs(params)
    local train_sx, train_sy
    local feval = function()
        neural_net_config.mlp:training()
        local batch_x, batch_y = batch_randomize(train_sx, train_sy, neural_net_config.training_batch_size)
        local cost = neural_net_config.criterion:forward(neural_net_config.mlp:forward(batch_x), batch_y)
        assert(tostring(cost) ~= 'nan' and not tostring(cost):find('inf'))
        grad_params:zero()
        neural_net_config.mlp:backward(batch_x, neural_net_config.criterion:backward(neural_net_config.mlp.output, batch_y))
        if neural_net_config.l1_weight and neural_net_config.l1_weight > 0 then
            for _, h in ipairs(neural_net_config.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * neural_net_config.l1_weight 
                h.gradWeight:add(neural_net_config.l1_weight, torch.sign(h.weight))
            end
        end
        if neural_net_config.l2_weight and neural_net_config.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * neural_net_config.l2_weight / 2 
            grad_params:add(neural_net_config.l2_weight, params)
        end
        if neural_net_config.max_grad and neural_net_config.max_grad > 0 then
            grad_params:clamp(-neural_net_config.max_grad, neural_net_config.max_grad)
        end
        return cost, grad_params 
    end
    local compute_cost = function(x, y)
        neural_net_config.mlp:evaluate()
        local cost = 0
        local batch_count = math.max(y:size(1)/neural_net_config.monitoring_batch_size, 1)
        for i = 1, batch_count do
            local batch_start = (i-1)*neural_net_config.monitoring_batch_size + 1
            local actual_batch_size = math.min(neural_net_config.monitoring_batch_size, 
                    y:size(1) - batch_start + 1)
            local batch_x = narrow_tensors(x, 1, batch_start, actual_batch_size)
            local batch_y = y:narrow(1, batch_start, actual_batch_size)
            local c = neural_net_config.criterion:forward(neural_net_config.mlp:forward(batch_x), batch_y)
            cost = cost + c * actual_batch_size
        end
        cost = cost / y:size(1)
        if neural_net_config.l1_weight and neural_net_config.l1_weight > 0 then
            for _, h in ipairs(neural_net_config.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * neural_net_config.l1_weight 
            end
        end
        if neural_net_config.l2_weight and neural_net_config.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * neural_net_config.l2_weight / 2 
        end
        return cost
    end 
    local optim_state = {
            learningRate = neural_net_config.learningRate, 
            learningRateDecay = neural_net_config.learningRateDecay,
            weightDecay = 0, 
            momentum = 0,
    }
    assert(optim_state.learningRate > 0)
    assert(optim_state.learningRateDecay >= 0)
    if initRange then
        params:uniform(-initRange, initRange)
    end
    for epoch_count = 1, neural_net_config.max_epochs do
        train_sx, train_sy = shuffle_data(train_x, train_y, train_sx, train_sy)
        for batch_no = 1, batch_count do
            adagrad_mod(feval, params, optim_state)
            
            example_count = example_count + neural_net_config.training_batch_size
            if neural_net_config.batch_reporting_frequency > 0 and
                    batch_no % neural_net_config.batch_reporting_frequency == 0 then
                local speed = example_count / (os.time()-start)
                io.write(string.format("Batch %d, speed = %.2f examples/s\r", batch_no, speed))
            end
        end
        if neural_net_config.batch_reporting_frequency > 0 and
                batch_count >= neural_net_config.batch_reporting_frequency then
            io.write('\n')
        end
        collectgarbage() -- important! avoid memory error

        if neural_net_config.epoch_reporting_frequency > 0 and 
                epoch_count % neural_net_config.epoch_reporting_frequency == 0 then
            print(string.format("Epoch %d:", epoch_count))
            print(string.format("\tTraining cost: %f", 
                    compute_cost(train_sx, train_sy)))
            if valid_x ~= nil and valid_y ~= nil then
                print(string.format("\tValidation cost: %f", 
                        compute_cost(valid_x, valid_y)))
            end
            print(string.format("\tTime elapsed: %d s", os.time()-start))
        end
        
        if valid_ds and neural_net_config.best_path then
            neural_net_config.mlp:evaluate()
            local uas = compute_UAS(valid_ds, parser, neural_net_config.monitoring_batch_size, neural_net_config.punc)
            io.write(string.format('Computed UAS on dev set: %.7f\n', uas))
            if best_uas == nil or uas > best_uas then
                io.write(string.format('Saving best model until now to %s... ', neural_net_config.best_path)) 
                torch.save(neural_net_config.best_path, neural_net_config.mlp)
                best_uas = uas
                print('Done.')
            end
        end
    end
    
    if valid_x ~= nil and valid_y ~= nil and neural_net_config.best_path ~= nil then
        io.write(string.format('Loading from %s... ', neural_net_config.best_path))
        neural_net_config.best_mlp = torch.load(neural_net_config.best_path)
        print('Done.')
    end
    local stop = os.time()
    print(string.format("Training %s... Done (%.2f min).", neural_net_config.model_name, (stop-start)/60.0))
end

local function pad_with_slash(s)
    if not s:endsWith('/') then
        return s .. '/'
    else
        return s
    end
end
    
function ptb_to_stanforddep(inp_dir_or_file, out_dir_or_file)
    print("in ptb_to_stanforddep")
    print(inp_dir_or_file)
    print(out_dir_or_file)
    if paths.filep(inp_dir_or_file) then
        create_directories(paths.dirname(out_dir_or_file))
        local cmd = 'java -cp stanford-parser-full-2013-11-12/stanford-parser.jar ' ..
                'edu.stanford.nlp.trees.EnglishGrammaticalStructure -maxLength 100 -basic -conllx ' ..
                '-treeFile ' .. inp_dir_or_file .. ' > ' .. out_dir_or_file .. '.dep'
        print(cmd) 
        assert(os.execute(cmd))
    else
        inp_dir_or_file = pad_with_slash(inp_dir_or_file)
        out_dir_or_file = pad_with_slash(out_dir_or_file) 
        for inp_path in search_files(inp_dir_or_file) do
            local out_path = inp_path:gsub(inp_dir_or_file:quote(), out_dir_or_file)
            ptb_to_stanforddep(inp_path, out_path)
        end
    end
end

function take_actions(rels, label_vocab, directed_labels)
    local actions = {
        vocab = Vocabulary(),
        left = {},
        right = {},
        rel2left = {},
        rel2right = {},
        action2rel = {},
        action2up = {},
        action2down = {},
    }
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