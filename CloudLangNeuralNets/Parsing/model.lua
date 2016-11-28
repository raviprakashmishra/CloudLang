require('torch')
require('data')
require('util')
require('PowConstant')
require('nn')
require('math')
require('optim')

local _LinearWithoutBias = torch.class('nn.LinearWithoutBias', 'nn.Linear')

    function _LinearWithoutBias:updateOutput(input)
       if input:dim() == 2 then
          local nframe = input:size(1)
          local nElement = self.output:nElement()
          self.output:resize(nframe, self.bias:size(1))
          if self.output:nElement() ~= nElement then
             self.output:zero()
          end
          self.output:addmm(0, self.output, 1, input, self.weight:t())
       else
          error('input must be matrix')
       end
       return self.output
    end

    function _LinearWithoutBias:accGradParameters(input, gradOutput, scale)
       scale = scale or 1
       if input:dim() == 2 then
          self.gradWeight:addmm(scale, gradOutput:t(), input)
       else
          error('input must be a matrix')
       end
    end
    
    function _LinearWithoutBias:parameters()
        return {self.weight}, {self.gradWeight}
    end
    
    
function new_cubenet(label_num, embedding_dims, embedding_num, hidden_num, 
        class_num, probabilistic, dropProb)
    local mlp = nn.Sequential()
    local lookup_table = nn.LookupTable(label_num, embedding_dims) 
    mlp:add(lookup_table)
    local dims = embedding_dims*embedding_num
    mlp:add(nn.View(-1, dims))
    if type(hidden_num) ~= 'table' then
        hidden_num = {hidden_num,}
    end
    local hidden_layers = {}
    local lastDims = dims
    for i = 1, #hidden_num do
        hidden_layers[i] = nn.Linear(lastDims, hidden_num[i]) 
        mlp:add(hidden_layers[i])
        mlp:add(nn.PowConstant(3))
        lastDims = hidden_num[i]
    end
    if dropProb then
        mlp:add(nn.Dropout(dropProb))
    end
    local output_layer = nn.LinearWithoutBias(lastDims, class_num)
    mlp:add(output_layer)
    if probabilistic then 
        mlp:add(nn.LogSoftMax())
    end
    mlp.get_lookup_table = function() return lookup_table end
    mlp.get_hidden_layers = function() return hidden_layers end
    mlp.get_output_layer = function() return output_layer end
    return mlp
end

function new_relunet(label_num, embedding_dims, embedding_num, hidden_num, 
        class_num, probabilistic, dropProb)
    local mlp = nn.Sequential()
    local lookup_table = nn.LookupTable(label_num, embedding_dims)
    mlp:add(lookup_table)
    local dims = embedding_dims*embedding_num
    mlp:add(nn.Reshape(dims, true))
    if type(hidden_num) ~= 'table' then
        hidden_num = {hidden_num,}
    end
    local hidden_layers = {}
    local lastDims = dims
    for i = 1, #hidden_num do
        hidden_layers[i] = nn.Linear(lastDims, hidden_num[i]) 
        mlp:add(hidden_layers[i])
        mlp:add(nn.ReLU(true))
        lastDims = hidden_num[i]
    end
    if dropProb then
        mlp:add(nn.Dropout(dropProb))
    end
    local output_layer = nn.LinearWithoutBias(lastDims, class_num)
    mlp:add(output_layer)
    if probabilistic then 
        mlp:add(nn.LogSoftMax())
    end
    mlp.get_lookup_table = function() return lookup_table end
    mlp.get_hidden_layers = function() return hidden_layers end
    mlp.get_output_layer = function() return output_layer end
    return mlp
end

local MaskLayer, Parent = torch.class('nn.MaskLayer', 'nn.Module')

function MaskLayer:__init(masks, filler)
    Parent.__init(self)
    self.filler = filler or 0
    self.masks = masks
    self.batch_masks = torch.ByteTensor() 
end

function MaskLayer:updateOutput(input)
    local data = input[1]
    local states = input[2]
    self.batch_masks:index(self.masks, 1, states)
    self.output:resizeAs(data):fill(self.filler)
    self.output[self.batch_masks] = data[self.batch_masks]
    return self.output
end

function MaskLayer:updateGradInput(input, gradOutput)
    self.gradInput = {gradOutput}
    return self.gradInput
end

function MaskLayer:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.filler)
end

function MaskLayer:type(type_, tensorCache)
    if type_ == 'torch.CudaTensor' then
        self.masks = self.masks:cuda()
        self.batch_masks = self.batch_masks:cuda() 
        self.output = self.output:cuda()
    else
        self.masks = self.masks:byte()
        self.batch_masks = self.batch_masks:byte()
        self.output = self.output:type(type_)
    end
end

function new_masked_net(core, masks, probabilistic)
    local mlp = nn.Sequential()
    local para = nn.ParallelTable()
    para:add(core)
    para:add(nn.Identity())
    mlp:add(para)
    mlp:add(nn.MaskLayer(masks, -math.huge))
    if probabilistic then 
        mlp:add(nn.LogSoftMax())
    end
    mlp.get_core = function() return core end
    mlp.get_lookup_table = function() return core:get_lookup_table() end
    mlp.get_hidden_layers = function() return core:get_hidden_layers() end
    mlp.get_output_layer = function() return core:get_output_layer() end
    return mlp
end

function shuffle2(x, y, sx, sy)
    local indices = torch.randperm(y:size(1)):long()
    if torch.isTensor(x) then
        if sx then
            sx:index(x, 1, indices)
        else
            sx = x:index(1, indices)
        end
    else -- assume it is a table of tensors
        if sx then
            for k, v in pairs(x) do
                sx[k]:index(x[k], 1, indices)
            end
        else
            sx = {}
            for k, v in pairs(x) do
                sx[k] = x[k]:index(1, indices)
            end
        end
    end
    if sy then
        sy:index(y, 1, indices)
    else
        sy = y:index(1, indices)
    end
    return sx, sy
end

function narrow_all(x, dim, start, size)
    if type(x) == 'table' then
        local ret = {}
        for k = 1, #x do
            ret[k] = x[k]:narrow(1, start, size)
        end
        return ret
    end
    return x:narrow(1, start, size)
end

function random_batch(x, y, batch_size)
    local batch_start = torch.random(math.max(y:size(1)-batch_size+1, 1))
    local actual_size = math.min(batch_size, y:size(1)-batch_start+1) 
    local batch_x = narrow_all(x, 1, batch_start, actual_size)
    local batch_y = y:narrow(1, batch_start, actual_size)
    return batch_x, batch_y 
end

function train(cfg, train_x, train_y, valid_x, valid_y, initRange)
    print(string.format("Training %s...", cfg.model_name))
    local start = os.time()
    local example_count = 0
    local batch_count = math.max(train_y:size(1)/cfg.training_batch_size, 1)
    local best_valid_cost = nil
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
            optim.adagrad(feval, params, optim_state)
--            optim.sgd(feval, params, optim_state)
            
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
        
        if valid_x and valid_y and cfg.best_path then
            local valid_cost = monitoring_cost(valid_x, valid_y)
            if best_valid_cost == nil or best_valid_cost > valid_cost then
                io.write(string.format('Writing best model to %s... ', cfg.best_path)) 
                torch.save(cfg.best_path, cfg.mlp)
                best_valid_cost = valid_cost
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
