require('paths')

function strip(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function find_files_recursively(dir)
  assert(dir and dir ~= "", "directory parameter is missing or empty")
  if string.sub(dir, -1) == "/" then
    dir = string.sub(dir, 1, -2)
  end

  local function yieldtree(p)
    if paths.dirp(p) then
        for entry in paths.files(p) do
            if entry ~= "." and entry ~= ".." then
                entry = p .. "/" .. entry
                yieldtree(entry)
            end
        end
     else
        coroutine.yield(p)
     end
  end

  return coroutine.wrap(function() yieldtree(dir) end)
end

function split(str, delimiter)
    -- imitate Python split function
   local result = {}
   local regex = string.format("([^%s]+)", delimiter)
   for part, _ in str:gmatch(regex) do
      table.insert(result, part)
   end
   return result 
end

function empty(t)
    for _, _ in pairs(t) do
        return false
    end
    return true
end

function string.startsWith(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function string.endsWith(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function defaultdict(default_value_factory)
    local mt = {
        __index = function(t, key)
            if not rawget(t, key) then
                rawset(t, key, default_value_factory())
            end
            return rawget(t, key)
        end
    }
    return setmetatable({}, mt)
end

function dict(a) 
    local ret = {}
    if a then
        for k, v in pairs(a) do
            ret[k] = v
        end
    end
    return ret
end

function len(a)
    local ret = 0
    for k, v in pairs(a) do
        ret = ret + 1
    end
    return ret
end

function list(a, b, c)
    local ret = {}
    if type(a) == 'table' then
        for k, v in pairs(a) do
            table.insert(ret, v)
        end
    elseif type(a) == 'function' then
        -- assume it is an iterator
        if b ~= nil or c ~= nil then
            for elem in a, b, c do
                table.insert(ret, elem)
            end
        else 
            for elem in a do
                table.insert(ret, elem)
            end
        end
    elseif a ~= nil or b ~= nil or c ~= nil then
        assert(false, "Unsupported parameter pattern")
    end
    return ret
end

function set(a, b, c)
    local ret = {}
    if type(a) == 'table' then
        for k, v in pairs(a) do
            ret[k] = true
        end
    elseif type(a) == 'function' then
        if b ~= nil or c ~= nil then
            for elem in a, b, c do
                ret[elem] = true
            end
        else 
            for elem in a do
                ret[elem] = true
            end
        end
    elseif a ~= nil or b ~= nil or c ~= nil then
        assert(false, "Unsupported parameter pattern")
    end
    return ret
end

function set2list(s)
    local ret = {}
    for k, v in pairs(s) do
        assert(v == true)
        table.insert(ret, k)
    end
    return ret
end

function range(from, to, step)
  step = step or 1
  if to == nil then
    to = from
    from = 1
  end
  return function(_, lastvalue)
    local nextvalue = lastvalue + step
    if step > 0 and nextvalue <= to or step < 0 and nextvalue >= to or
       step == 0
    then
      return nextvalue
    end
  end, nil, from - step
end

function map(func, tbl)
    local newtbl = {}
    for i,v in pairs(tbl) do
        newtbl[i] = func(v)
    end
    return newtbl
end

function list_iter(t)
    if type(t) == 'function' then
        return t
    end
    local i = 0
    local n = table.getn(t)
    return function ()
               i = i + 1
               if i <= n then return t[i] end
           end
end

function arg_iter(...)
    return list_iter({...})
end

-- chain an iterator of iterators
function iter_chain2(iter_of_iters)
    local function yield_values()
        for iter in iter_of_iters do
            for v in iter do
                coroutine.yield(v)
            end
        end
    end
    return coroutine.wrap(yield_values) 
end

-- chain a variable number of iterators
function iter_chain(...)
    return iter_chain2(list_iter({...})) 
end

function iter_zip(...)
    local iters = {...} 
    local function yield_values()
        local values = {}
        while true do
            for i, iter in ipairs(iters) do
                values[i] = iter()
                if not values[i] then
                    return
                end
            end
            coroutine.yield(table.unpack(values))
        end
    end
    return coroutine.wrap(yield_values) 
end

function iter_range(from, to, step)
    if to == nil then
        to = from
        from = 1
    end
    step = step or 1    
    local function yield_values()
        for v = from, to, step do
            coroutine.yield(v)
        end
    end
    return coroutine.wrap(yield_values) 
end

function iter_map(func, iter)
    local function yield_values()
        for v in iter do
            coroutine.yield(func(v))
        end
    end
    return coroutine.wrap(yield_values) 
end

function iter_size(it)
    local count = 0
    for _ in it do
        count = count + 1
    end
    return count
end

function iter_first_n(it, n)
    local function yield_values()
        local count = 0
        for v in it do
            count = count + 1
            if count > n then
                return
            else
                coroutine.yield(v)
            end
        end
    end
    return coroutine.wrap(yield_values) 
end

function enumerate(it)
    local function yield_values()
        local count = 0
        for v in it do
            count = count + 1
            coroutine.yield(count, v)
        end
    end
    return coroutine.wrap(yield_values) 
end

iter_len = iter_size

function to_tensor(x, y, cuda_enabled)
    -- check first...
    if x ~= nil then
        for i = 1, #x do
            assert(#x[i] == #x[1])
        end
        if y ~= nil then
            assert(#y == #x[1])
        end
    end
    -- and convert
    if x ~= nil then
        for i = 1, #x do
            if cuda_enabled then
                x[i] = torch.CudaTensor(x[i])
            else
                x[i] = torch.LongTensor(x[i])
            end
        end
    end
    if y ~= nil then
        if cuda_enabled then
            y = torch.CudaTensor(y)
        else
            y = torch.LongTensor(y)
        end
    end
    return x, y
end

function ll_insert(list, head, index)
    if head == 0 then
        return index
    end
    if index < head then
        list[index] = head
        return index
    end
    local prev = 0
    local curr = head
    while curr > 0 and curr < index do
        prev = curr
        curr = list[curr]
    end
    list[prev] = index
    list[index] = curr
    return head
end

function ll_get(list, head, list_index)
    local curr = head
    while curr > 0 and list_index > 1 do
        curr = list[curr]
        list_index = list_index - 1 
    end
    if curr > 0 then
        return curr
    else
        return nil
    end
end

function ll_rget(list, head, list_index)
    local indices = {}
    local curr = head
    while curr > 0 do
        table.insert(indices, curr)
        curr = list[curr]
    end
    return indices[#indices-list_index+1]
end

function mkdirs(path)
    if path == '/' then return true end
    local parent = paths.dirname(path)
    if not paths.dirp(parent) then
        if not mkdirs(parent) then
            return false
        end
    end
    return paths.mkdir(path)
end

function ternary(condition, if_true, if_false)
    if condition then 
        return if_true 
    else 
        return if_false 
    end
end

local quotepattern = '(['..("%^$().[]*+-?"):gsub("(.)", "%%%1")..'])'
string.quote = function(str)
    return str:gsub(quotepattern, "%%%1")
end