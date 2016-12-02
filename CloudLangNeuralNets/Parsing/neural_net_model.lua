require('torch')
require('util')
require('nn')
require('math')
require('optim')

local PowConstant, parent = torch.class('nn.PowConstant', 'nn.Module')

    function PowConstant:__init(constant_scalar)
    parent.__init(self)
    assert(type(constant_scalar) == 'number', 'Power needs to be scalar')
    assert(constant_scalar > 1, 'Power should be > 1')
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
          error('Wrong input')
       end
       return self.output
    end

    function _LinearWithoutBias:accGradParameters(input, gradOutput, scale)
       scale = scale or 1
       if input:dim() == 2 then
          self.gradWeight:addmm(scale, gradOutput:t(), input)
       else
          error('Wrong Input')
       end
    end
    
    function _LinearWithoutBias:parameters()
        return {self.weight}, {self.gradWeight}
    end
    
    
function transition_parser_neuralnet(label_num, embedding_dims, embedding_num, hidden_num, 
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

function mask_neural_net(core, masks, probabilistic)
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

function shuffle_data(x, y, sx, sy)
    local indices = torch.randperm(y:size(1)):long()
    if torch.isTensor(x) then
        if sx then
            sx:index(x, 1, indices)
        else
            sx = x:index(1, indices)
        end
    else
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

function narrow_tensors(x, dim, start, size)
    if type(x) == 'table' then
        local ret = {}
        for k = 1, #x do
            ret[k] = x[k]:narrow(1, start, size)
        end
        return ret
    end
    return x:narrow(1, start, size)
end

function batch_randomize(x, y, batch_size)
    local batch_start = torch.random(math.max(y:size(1)-batch_size+1, 1))
    local actual_size = math.min(batch_size, y:size(1)-batch_start+1) 
    local batch_x = narrow_tensors(x, 1, batch_start, actual_size)
    local batch_y = y:narrow(1, batch_start, actual_size)
    return batch_x, batch_y 
end

function train_model(nn_config, train_x, train_y, valid_x, valid_y, initRange)
    print(string.format("Training %s...", nn_config.model_name))
    local start = os.time()
    local example_count = 0
    local batch_count = math.max(train_y:size(1)/nn_config.training_batch_size, 1)
    local best_valid_cost = nil
    local params, grad_params = nn_config.mlp:getParameters()
    local params2 = torch.Tensor():typeAs(params)
    local train_sx, train_sy
    local feval = function()
        nn_config.mlp:training()
        local batch_x, batch_y = batch_randomize(train_sx, train_sy, nn_config.training_batch_size)
        local cost = nn_config.criterion:forward(nn_config.mlp:forward(batch_x), batch_y)
        assert(tostring(cost) ~= 'nan' and not tostring(cost):find('inf'))
        grad_params:zero()
        nn_config.mlp:backward(batch_x, nn_config.criterion:backward(nn_config.mlp.output, batch_y))
        if nn_config.l1_weight and nn_config.l1_weight > 0 then
            for _, h in ipairs(nn_config.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * nn_config.l1_weight 
                h.gradWeight:add(nn_config.l1_weight, torch.sign(h.weight))
            end
        end
        if nn_config.l2_weight and nn_config.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * nn_config.l2_weight / 2 
            grad_params:add(nn_config.l2_weight, params)
        end
        if nn_config.max_grad and nn_config.max_grad > 0 then
            grad_params:clamp(-nn_config.max_grad, nn_config.max_grad)
        end
        return cost, grad_params 
    end
    local compute_cost = function(x, y)
        nn_config.mlp:evaluate()
        local cost = 0
        local batch_count = math.max(y:size(1)/nn_config.monitoring_batch_size, 1)
        for i = 1, batch_count do
            local batch_start = (i-1)*nn_config.monitoring_batch_size + 1
            local actual_batch_size = math.min(nn_config.monitoring_batch_size, 
                    y:size(1) - batch_start + 1)
            local batch_x = narrow_tensors(x, 1, batch_start, actual_batch_size)
            local batch_y = y:narrow(1, batch_start, actual_batch_size)
            local c = nn_config.criterion:forward(nn_config.mlp:forward(batch_x), batch_y)
            cost = cost + c * actual_batch_size
        end
        cost = cost / y:size(1)
        if nn_config.l1_weight and nn_config.l1_weight > 0 then
            for _, h in ipairs(nn_config.mlp:get_hidden_layers()) do
                cost = cost + torch.abs(h.weight):sum() * nn_config.l1_weight 
            end
        end
        if nn_config.l2_weight and nn_config.l2_weight > 0 then
            cost = cost + params2:pow(params, 2):sum() * nn_config.l2_weight / 2 
        end
        return cost
    end 
    local optim_state = {
            learningRate = nn_config.learningRate, 
            learningRateDecay = nn_config.learningRateDecay,
            weightDecay = 0, 
            momentum = 0,
    }
    assert(optim_state.learningRate > 0)
    assert(optim_state.learningRateDecay >= 0)
    if initRange then
        params:uniform(-initRange, initRange)
    end
    for epoch_count = 1, nn_config.max_epochs do
        train_sx, train_sy = shuffle_data(train_x, train_y, train_sx, train_sy)
        for batch_no = 1, batch_count do
            optim.adagrad(feval, params, optim_state)
            
            example_count = example_count + nn_config.training_batch_size
            if nn_config.batch_reporting_frequency > 0 and
                    batch_no % nn_config.batch_reporting_frequency == 0 then
                local speed = example_count / (os.time()-start)
                io.write(string.format("Batch %d, speed = %.2f examples/s\r", batch_no, speed))
            end
        end
        if nn_config.batch_reporting_frequency > 0 and
                batch_count >= nn_config.batch_reporting_frequency then
            io.write('\n')
        end
        collectgarbage()

        if nn_config.epoch_reporting_frequency > 0 and 
                epoch_count % nn_config.epoch_reporting_frequency == 0 then
            print(string.format("Epoch %d:", epoch_count))
            print(string.format("\tTraining cost: %f", 
                    compute_cost(train_sx, train_sy)))
            if valid_x ~= nil and valid_y ~= nil then
                print(string.format("\tValidation cost: %f", 
                        compute_cost(valid_x, valid_y)))
            end
            print(string.format("\tTime elapsed: %d s", os.time()-start))
        end
        
        if valid_x and valid_y and nn_config.best_path then
            local valid_cost = compute_cost(valid_x, valid_y)
            if best_valid_cost == nil or best_valid_cost > valid_cost then
                io.write(string.format('Saving best model until now to %s... ', nn_config.best_path)) 
                torch.save(nn_config.best_path, nn_config.mlp)
                best_valid_cost = valid_cost
                print('Done.')
            end
        end
    end
    if valid_x ~= nil and valid_y ~= nil and nn_config.best_path ~= nil then
        io.write(string.format('Loading from %s... ', nn_config.best_path))
        nn_config.best_mlp = torch.load(nn_config.best_path)
        print('Done.')
    end
    local stop = os.time()
    print(string.format("Training %s... Done (%.2f min).", nn_config.model_name, (stop-start)/60.0))
end
