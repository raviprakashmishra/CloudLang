require('paths')
require('torch')

function search_files(dir)
  assert(dir and dir ~= "", "directory not provided")
  if string.sub(dir, -1) == "/" then
    dir = string.sub(dir, 1, -2)
  end

  local function find_dir_tree(p)
    if paths.dirp(p) then
        for entry in paths.files(p) do
            if entry ~= "." and entry ~= ".." then
                entry = p .. "/" .. entry
                find_dir_tree(entry)
            end
        end
     else
        coroutine.yield(p)
     end
  end

  return coroutine.wrap(function() find_dir_tree(dir) end)
end

function split_string(str, delimiter)
   local result = {}
   local regex = string.format("([^%s]+)", delimiter)
   for part, _ in str:gmatch(regex) do
      table.insert(result, part)
   end
   return result 
end

function string.endsWith(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end

function table_length(a)
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
        assert(false, "One or more parameters empty or nil")
    end
    return ret
end

function find_range(from, to, step)
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

function iterate_list(t)
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

function chain_iterators(iter_of_iters)
    local function yield_values()
        for iter in iter_of_iters do
            for v in iter do
                coroutine.yield(v)
            end
        end
    end
    return coroutine.wrap(yield_values) 
end

function iterate_thru_range(from, to, step)
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

function map_iterator(func, iter)
    local function yield_values()
        for v in iter do
            coroutine.yield(func(v))
        end
    end
    return coroutine.wrap(yield_values) 
end

function size_of_iterator(it)
    local count = 0
    for _ in it do
        count = count + 1
    end
    return count
end

iter_len = size_of_iterator

function insert_linked_list(list, head, index)
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

function search_linked_list(list, head, list_index)
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

function search_next_linked_list(list, head, list_index)
    local indices = {}
    local curr = head
    while curr > 0 do
        table.insert(indices, curr)
        curr = list[curr]
    end
    return indices[#indices-list_index+1]
end

function create_directories(path)
    if path == '/' then return true end
    local parent = paths.dirname(path)
    if not paths.dirp(parent) then
        if not create_directories(parent) then
            return false
        end
    end
    return paths.mkdir(path)
end

function ops_ternary(condition, if_true, if_false)
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

local max_words = 50
local function parseString(file)  
    local str = {}
    for i = 1, max_words do
        local char = file:readChar()
        if char == 32 or char == 10 or char == 0 then
            break
        else
            str[#str+1] = char
        end
    end
    str = torch.CharStorage(str)
    return str:string()
end

function get_max_vocab_index(vocab)
    if torch.type(vocab) == 'Vocabulary' then
        return vocab.indexer:max()
    else
        local indexer
        for _, v in pairs(vocab) do
            assert(torch.type(v) == 'Vocabulary')
            if indexer == nil then
                indexer = v.indexer
            end
            assert(indexer == v.indexer)
        end
        return indexer:max()
    end
end

local _Indexer = torch.class('Indexer')

    function _Indexer:__init()
        self.index = 0
    end
    
    function _Indexer:max()
        return self.index
    end
    
    function _Indexer:next()
        self.index = self.index + 1
        return self.index
    end

local _Vocabulary = torch.class('Vocabulary')
    
    function _Vocabulary:__init(indexer)
        self.indexer = indexer or Indexer()
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.sealed = false
    end
    
    function _Vocabulary:add_all(v)
        for word, _ in pairs(v.word2index) do
            self:get_index(word)
        end
    end
        
    function _Vocabulary:get_index(word)
        word = word or '__NONE__'
        if self.word2index[word] then
            self.word2count[word] = self.word2count[word] + 1
        else
            if self.sealed then
                word = '__MISSING__'
            else
                local index = self.indexer:next()
                self.word2index[word] = index
                self.index2word[index] = word
                self.word2count[word] = 1
            end
        end
        return self.word2index[word]
    end
    
    function _Vocabulary:get_word(index)
        return self.index2word[index]
    end
    
    function _Vocabulary:prune(min_count, new_indexer)
        assert(not self.sealed)
        self.indexer = new_indexer or Indexer()
        for word, count in pairs(self.word2count) do
            if count >= min_count then
                local index = self.indexer:next()
                self.word2index[word] = index
                self.index2word[index] = word
                self.word2count[word] = count
            else
                -- remove
                self.word2index[word] = nil
                self.word2count[word] = nil
            end
        end
    end

    function _Vocabulary:seal(b)
        self.sealed = b
    end
    
    function _Vocabulary:size()
        return table_length(self.word2index)
    end

local function get_label(rel)
    rel = rel:gsub('#.+$', '')
    return rel
end

local _Format_Conll = torch.class('Format_Conll')

    function _Format_Conll:__init(cutoff_frequency, lemma_enabled, directed_lables)
        self.cutoff_frequency = cutoff_frequency or 5
        self.lemma_enabled = lemma_enabled
        self.directed_lables = directed_lables

        local indexer = Indexer()
        self.vocabs = {
            word = Vocabulary(indexer),
            pos = Vocabulary(indexer),
            label = Vocabulary(indexer),
        }
        self.vocabs.word:get_index('__MISSING__')
        self.vocabs.word:get_index('__NONE__')
        self.vocabs.pos:get_index('__MISSING__')
        self.vocabs.pos:get_index('__NONE__')
        self.vocabs.label:get_index('__NONE__')
        if self.lemma_enabled then
            vocabs.lemma = Vocabulary(indexer)
            self.vocabs.lemma:get_index('__MISSING__')
            self.vocabs.lemma:get_index('__NONE__')
        end
    end

    function _Format_Conll:process_data(path_iter, name, max_rows)
        print('Initializing data processor... ')
        self:compile_dataset(path_iter, name, max_rows)
        if self.cutoff_frequency > 1 then
            local indexer = Indexer()
            for _, vocab in pairs(self.vocabs) do
                vocab:prune(self.cutoff_frequency, indexer)
            end
        end
        for _, vocab in pairs(self.vocabs) do
            vocab:seal(true)
        end
        print('Initializing data processor... Done.')
    end
    
    function _Format_Conll:compile_dataset(path_iter, name, max_rows) 
        name = name or '[noname]'
        print(string.format("Compiling dataset '%s'... ", name))
        start = os.time()
        local sents = torch.LongTensor(max_rows, 3)
        local tokens
        if self.lemma_enabled then
            tokens = torch.LongTensor(max_rows, 6)
        else
            tokens = torch.LongTensor(max_rows, 5)
        end
        local sent_count = 0
        local token_count = 0
        if type(path_iter) == 'string' then
            path_iter = iterate_list({path_iter})
        end
        for path in path_iter do
            for lines in self:iterate_sentences(path) do
                sent_count = sent_count + 1
                sents[sent_count][1] = sent_count
                sents[sent_count][2] = token_count + 1
                token_count = self:parse_sentence(lines, tokens, token_count)
                sents[sent_count][3] = token_count - sents[sent_count][2] + 1
             end
        end
        sents = sents:narrow(1, 1, sent_count)
        tokens = tokens:narrow(1, 1, token_count)
        collectgarbage()
        stop = os.time()
        print(string.format("Compiling dataset '%s'... Done (%d tokens, %d sentences, %d s).", 
                name, token_count, sent_count, stop-start))
        return {['sents'] = sents, ['tokens'] = tokens}
    end
    
    function _Format_Conll:iterate_sentences(path)
        local function yield_sentences() 
            local f = io.open(path, 'r')
            local line = f:read('*line')
            while line do
                local lines = {}
                while line do
                    if line == '' then break end
                    if line:sub(1,1) ~= '#' then
                        table.insert(lines, line)
                    end
                    line = f:read('*line')
                end
                if #lines > 0 then
                    coroutine.yield(lines)
                end
                line = f:read('*line')
            end
            f:close()
        end
        return coroutine.wrap(yield_sentences)
    end
    
    function _Format_Conll:parse_sentence(lines, tokens, token_count)
        token_count = token_count + 1
        tokens[{token_count, 1}] = self.vocabs.word:get_index('__ROOT__')
        tokens[{token_count, 2}] = self.vocabs.pos:get_index('__ROOT__')
        tokens[{token_count, 3}] = 0
        tokens[{token_count, 4}] = self.vocabs.label:get_index('__NONE__')
        tokens[{token_count, 5}] = self.vocabs.label:get_index('__NONE__')
        if self.lemma_enabled then
            tokens[token_count][6] = self.vocabs.lemma:get_index('__ROOT__')
        end
        for _, line in ipairs(lines) do
            token_count = token_count + 1
            local fields = split_string(line, '\t')
            tokens[{token_count, 1}] = self.vocabs.word:get_index(fields[2])
            if self.lemma_enabled then
                tokens[{token_count, 6}] = self.vocabs.lemma:get_index(fields[3])
            end
            tokens[{token_count, 2}] = self.vocabs.pos:get_index(fields[4]) 
            if fields[7] ~= '_' then
                field_num = tonumber(fields[7])
                if field_num == nil then
                    field_num = tonumber(fields[6])
                end
                head_id = field_num + 1
                tokens[token_count][3] = head_id
                if self.directed_lables then
                    tokens[{token_count, 4}] = self.vocabs.label:get_index(fields[8] .. "#U")
                    tokens[{token_count, 5}] = self.vocabs.label:get_index(fields[8] .. "#D")
                else
                    local f = self.vocabs.label:get_index(fields[8])
                    assert(f, "Unknown label: " .. fields[8])
                    tokens[{token_count, 4}] = f
                    tokens[{token_count, 5}] = f
                end
            end
        end
        return token_count
    end
    
    function _Format_Conll:write_sentence_conll_format(f, tokens)
        for i = 2, tokens:size(1) do
            local word = self.vocabs.word:get_word(tokens[i][1])
            local cpos = self.vocabs.pos:get_word(tokens[i][2])
            local lemma = '_'
            if self.lemma_enabled then
                lemma = self.vocabs.lemma:get_word(tokens[i][6])
            end
            local head = 1000 -- dummy value
            local head_label = '_'
            if tokens[i][3] > 0 then
                head = tokens[i][3] - 1
                head_label = get_label(self.vocabs.label:get_word(tokens[i][4]))
            end
            f:write(string.format('%d\t%s\t%s\t%s\t_\t_\t%d\t%s\t_\t_',
                    i-1, word, lemma, cpos, head, head_label))
            f:write('\n')
        end
    end
    
    function _Format_Conll:write_sentence(file_or_path, tokens)
        if type(file_or_path) == 'string' then
            local f = io.open(file_or_path, 'w') 
            self:write_sentence_conll_format(f, tokens)
            f:close()
        else
            self:write_sentence_conll_format(file_or_path, tokens)
        end
    end
    
    function _Format_Conll:replace_dep_rel(input, output, ds)
        local f = io.open(output, 'w')
        local s = 0
        for lines in self:iterate_sentences(input) do
            s = s + 1
            if s > ds.sents:size(1) then return end
            local t = ds.sents[s][2]
            for _, line in ipairs(lines) do
                t = t + 1
                local fields = split_string(line, '\t')
                if ds.tokens[t][3] > 0 then
                    fields[7] = tostring(ds.tokens[t][3] - 1)
                    local rel_name = self.vocabs.label:get_word(ds.tokens[t][4])
                    if not rel_name then
                        error("Unknown relation index: " .. ds.tokens[t][4])
                    end
                    fields[8] = get_label(rel_name)
                else
                    fields[7] = ''
                    fields[8] = ''
                end
                f:write(table.concat(fields, '\t'))                
                f:write('\n')
            end
            assert(t == ds.sents[s][2] + ds.sents[s][3] - 1)
            f:write('\n')
        end
        assert(s == ds.sents:size(1))
        f:close()
    end

function is_tree_valid(heads)
    local h = {}
    table.insert(h, -1)
    for i = 1, heads:size(1) do
        if heads[i] < 0 or heads[i] > heads:size(1) then
            return false
        end
        table.insert(h, -1)
    end
    for i = 1, heads:size(1) do
        local k = i
        while k > 0 do
            if h[k] >= 0 and h[k] < i then
                break
            end
            if h[k] == i then
                return false
            end
            h[k] = i
            k = heads[k]
        end
    end
    return true
end

function is_tree_projective(heads)
    if not is_tree_valid(heads) then
        return false
    end

    local counter = 0
    local visit_tree
    visit_tree = function(w)
        for i = 1, w-1 do
            if heads[i] == w and not visit_tree(i) then
                return false
            end
        end
        counter = counter + 1;
        if w ~= counter then
            return false
        end
        for i = w+1, heads:size(1) do
            if heads[i] == w and not visit_tree(i) then
                return false
            end
        end
        return true
    end
    return visit_tree(1)
end

function left_dependency(links, wid, index)
    index = index or 1
    assert(index > 0)
    if wid == nil then
        return nil
    end
    if links:size(2) == 5 then
        local ret = search_linked_list(links:select(2, 5), links[wid][4], index)
        return ops_ternary(ret and ret < wid, ret, nil)
    else
        for i = 1, wid-1 do
            if links[{i, 1}] == wid then
                index = index - 1
            end
            if index == 0 then
                return i
            end
        end
        return nil
    end
end

function right_dependency(links, wid, index)
    index = index or 1
    assert(index > 0)
    if wid == nil then
        return nil
    end
    if links:size(2) == 5 then
        local ret = search_next_linked_list(links:select(2, 5), links[wid][4], index)
        return ops_ternary(ret and ret > wid, ret, nil)
    else
        for i = links:size(1), wid+1, -1 do
            if links[{i, 1}] == wid then
                index = index - 1
            end
            if index == 0 then
                return i
            end
        end
        return nil
    end
end
