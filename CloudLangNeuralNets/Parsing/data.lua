require('util')
require('torch')

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
        return len(self.word2index)
    end

function max_index(vocab)
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
    
local max_w = 50

local function readStringv2(file)  
    local str = {}
    for i = 1, max_w do
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

function read_vectors_from_text_file(path, normalized, separator)
    separator = separator or '\t'
    print(string.format('Reading word vector file at %s... ', path))
    io.write('Counting... ')
    local rows = iter_len(io.lines(path))+1
    print(string.format('Done (%d rows).', rows))
    local file = io.open(path, 'r')
    local line = file:read('*line')
    local cols = #split(line, separator)-1
    local M = torch.Tensor(rows, cols)
    local vocab = Vocabulary()
    M[vocab:get_index('__MISSING__')]:uniform() -- random vector for __MISSING__
    local count = 0
    while line ~= nil do
        local fields = split(line, separator)
        local word = fields[1] 
        for j = 2, #fields do
            M[vocab:get_index(word)][j-1] = tonumber(fields[j])
        end
        count = count + 1
        if count % 100000 == 0 then
            print(string.format('%d ...', count))
        end
        line = file:read('*line')
    end
    file:close()
    if normalized then
        local norms = M:norm(2, 2)
        norms[norms:eq(0)] = 1
        M:cdiv(norms:expand(rows, cols))
    end
    print(string.format('Reading word vector file at %s... Done.\n', path))
    
    return vocab, M
end


function read_word2vec_bin(path, normalized, new_format)
    print(string.format('Reading word2vec binary file at %s... ', path))
    local file = torch.DiskFile(path, 'r')
    
    --Reading Header
    file:ascii()
    local words = file:readInt()
    local size = file:readInt()
    print(string.format('%d word, %d dimensions each', words, size))
    
    local vocab = Vocabulary()
    local M = torch.FloatTensor(words+1, size)
    M[vocab:get_index('__MISSING__')]:uniform() -- random vector for __MISSING__

    --Reading Contents
    file:binary()
    for i = 2, words+1 do
        local str = readStringv2(file)
        local vecrep = file:readFloat(size)
        vecrep = torch.FloatTensor(vecrep)
        M:select(1, i):copy(vecrep)
        vocab.word2index[str] = i
        vocab.word2count[str] = 1
        if new_format then
            local char = file:readChar()
            assert(char == 32 or char == 10 or char == 0)
        end
        if i % 100000 == 0 then
            print(string.format('%d ...', i))
        end
    end
    file:close()
    if normalized then
        local norms = M:norm(2, 2)
        norms[norms:eq(0)] = 1
        M:cdiv(norms:expand(words+1, size))
    end
    print('Done.\n')
    
    return vocab, M
end

function kfold_interleaving(iter_generator, k)
    -- folds alternate and repeat: 0,1,2,...,9,0,1,...
    local training = function(index)
        local count = 0
        for elem in iter_generator() do
            if count % k ~= index then
                coroutine.yield(elem)
            end
            count = count + 1
        end
    end
    local testing = function(index)
        local count = 0
        for elem in iter_generator() do
            if count % k == index then
                coroutine.yield(elem)
            end
            count = count + 1
        end
    end
    local folds = {}
    for index = 0, k-1 do
        table.insert(folds, {
                train = function() return coroutine.wrap(function() return training(index) end) end,
                test = function() return coroutine.wrap(function() return testing(index) end) end,
        })
    end
    return folds
end

function kfold_contiguous(iter_generator, k, size, gen_valid)
    -- folds form contiguous blocks: 0,0,...0,1,1,...,1,...
    size = size or iter_size(iter_generator())
    local block_size = math.ceil(size / k)
    local training = function(index)
        local count = 0
        for elem in iter_generator() do
            local block_no = math.floor(count / block_size)
            if block_no ~= index and
                    (not gen_valid or block_no ~= (index % 10 + 1)) then
                coroutine.yield(elem)
            end
            count = count + 1
        end
    end
    local testing = function(index)
        local count = 0
        for elem in iter_generator() do
            local block_no = math.floor(count / block_size)
            if block_no == index then
                coroutine.yield(elem)
            end
            if block_no > index then
                break
            end
            count = count + 1
        end
    end
    local valid = function(index)
        if not gen_valid then
            return
        end
        local count = 0
        for elem in iter_generator() do
            local block_no = math.floor(count / block_size)
            if block_no == (index % 10 + 1) then
                coroutine.yield(elem)
            end
            if block_no > (index % 10 + 1) then
                break
            end
            count = count + 1
        end
    end
    local folds = {}
    for index = 0, k-1 do
        table.insert(folds, {
                train = function() return coroutine.wrap(function() return training(index) end) end,
                valid = function() return coroutine.wrap(function() return valid(index) end) end,
                test = function() return coroutine.wrap(function() return testing(index) end) end,
        })
    end
    return folds
end

local function get_label(rel)
    rel = rel:gsub('#.+$', '') -- avoid returning substitution count
    return rel
end

local _CoNLL = torch.class('CoNLL')

    function _CoNLL:__init(cutoff_frequency, lemma_enabled, directed_lables)
        print("Initialize CoNLL")
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
            print("CoNLL init in lemma_enabled")
            vocabs.lemma = Vocabulary(indexer)
            self.vocabs.lemma:get_index('__MISSING__')
            self.vocabs.lemma:get_index('__NONE__')
        end
    end

    function _CoNLL:prepare(path_iter, name, max_rows)
        print('Preparing dataset builder... ')
        self:build_dataset(path_iter, name, max_rows)
        if self.cutoff_frequency > 1 then
            local indexer = Indexer()
            for _, vocab in pairs(self.vocabs) do
                vocab:prune(self.cutoff_frequency, indexer)
            end
        end
        for _, vocab in pairs(self.vocabs) do
            vocab:seal(true)
        end
        print('Preparing dataset builder... Done.')
    end
    
    function _CoNLL:build_dataset(path_iter, name, max_rows) 
        name = name or '[noname]'
        print(string.format("Building dataset '%s'... ", name))
        start = os.time()
        local sents = torch.LongTensor(max_rows, 3)
        local tokens
        if self.lemma_enabled then
            print("CoNLL build_dataset in lemma_enabled")
            tokens = torch.LongTensor(max_rows, 6)
        else
            tokens = torch.LongTensor(max_rows, 5)
        end
        local sent_count = 0
        local token_count = 0
        if type(path_iter) == 'string' then
            print("CoNLL path_iter is a string")
            path_iter = list_iter({path_iter})
        end
        for path in path_iter do
            print("path " .. path)
            for lines in self:iter_sentences(path) do
                sent_count = sent_count + 1
                print("sent_count " .. sent_count)
                sents[sent_count][1] = sent_count
                print("sents[sent_count][1] " .. sents[sent_count][1])
                sents[sent_count][2] = token_count + 1
                print("sents[sent_count][2] " .. sents[sent_count][2])
                token_count = self:parse_sentence(lines, tokens, token_count)
                print("token_count " .. token_count)
                sents[sent_count][3] = token_count - sents[sent_count][2] + 1
                print("sents[sent_count][3] " .. sents[sent_count][3])
             end
        end
        print("token tensor")
        print(tokens[{2, 1}])
        print(tokens[{2, 2}])
        print(tokens[{2, 3}])
        print(tokens[{2, 4}])
        print(tokens[{2, 5}])
        sents = sents:narrow(1, 1, sent_count)
        tokens = tokens:narrow(1, 1, token_count)
        collectgarbage() -- important! avoid memory error
        stop = os.time()
        print(string.format("Building dataset '%s'... Done (%d tokens, %d sentences, %d s).", 
                name, token_count, sent_count, stop-start))
        return {['sents'] = sents, ['tokens'] = tokens}
    end
    
    function _CoNLL:iter_sentences(path)
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
    
    function _CoNLL:parse_sentence(lines, tokens, token_count)
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
            --print("line " .. line)
            token_count = token_count + 1
            local fields = split(line, '\t')
            tokens[{token_count, 1}] = self.vocabs.word:get_index(fields[2])
            if self.lemma_enabled then
                tokens[{token_count, 6}] = self.vocabs.lemma:get_index(fields[3])
            end
            tokens[{token_count, 2}] = self.vocabs.pos:get_index(fields[4]) 
            if fields[7] ~= '_' then
                head_id = tonumber(fields[7]) + 1
                tokens[token_count][3] = head_id
                if self.directed_lables then
                    tokens[{token_count, 4}] = self.vocabs.label:get_index(fields[8] .. "#U")
                    tokens[{token_count, 5}] = self.vocabs.label:get_index(fields[8] .. "#D")
                else
                    local f = self.vocabs.label:get_index(fields[8])
                    assert(f, "Strange label: " .. fields[8])
                    tokens[{token_count, 4}] = f
                    tokens[{token_count, 5}] = f
                end
            end
        end
        return token_count
    end
    
    function _CoNLL:_write_sentence_to_file(f, tokens)
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
    
    function _CoNLL:write_sentence(file_or_path, tokens)
        if type(file_or_path) == 'string' then
            local f = io.open(file_or_path, 'w') 
            self:_write_sentence_to_file(f, tokens)
            f:close()
        else
            self:_write_sentence_to_file(file_or_path, tokens)
        end
    end
    
    function _CoNLL:write_all_sentences(path, ds)
        local f = io.open(path, 'w')
        for s = 1, ds.sents:size(1) do
            local tokens = ds.tokens:narrow(1, ds.sents[s][2], ds.sents[s][3])
            self:_write_sentence_to_file(f, tokens)
            f:write('\n')
        end
        f:close()
    end
    
    function _CoNLL:substitue_dependency(input, output, ds)
        --[[
        Read a CoNLL file in `input`, replace all dependency links with those
        specified by `ds` and write to `output`.
        --]]
        local f = io.open(output, 'w')
        local s = 0
        for lines in self:iter_sentences(input) do
            s = s + 1
            if s > ds.sents:size(1) then return end
            local t = ds.sents[s][2]
            for _, line in ipairs(lines) do
                t = t + 1
                local fields = split(line, '\t')
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

function penn_sentences(path)
    local function yield_sentences()
        local bracket_count = 0
        local sentence = {}
        for line in io.lines(path) do
            table.insert(sentence, strip(line))
            bracket_count = bracket_count + iter_size(line:gfind('%('))
            bracket_count = bracket_count - iter_size(line:gfind('%)'))
            if bracket_count == 0 then
                sentence = strip(table.concat(sentence, ' '))
                if sentence ~= '' then
                    coroutine.yield(sentence)
                end
                sentence = {}
            end
        end
    end
    return coroutine.wrap(yield_sentences)
end

-- check if the tree is legal, O(n)
function is_tree(heads)
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

function is_projective(heads)
    if not is_tree(heads) then
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

function cget_left_dependent(links, wid, index)
    index = index or 1
    assert(index > 0)
    if wid == nil then
        return nil
    end
    if links:size(2) == 5 then -- linked list exists
        -- 4th column is leftmost child, 5th is siblings
        local ret = ll_get(links:select(2, 5), links[wid][4], index)
        return ternary(ret and ret < wid, ret, nil)
    else -- linked list doesn't exists
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

function cget_right_dependent(links, wid, index)
    index = index or 1
    assert(index > 0)
    if wid == nil then
        return nil
    end
    if links:size(2) == 5 then -- linked list exists
        -- 4th column is leftmost child, 5th is siblings
        local ret = ll_rget(links:select(2, 5), links[wid][4], index)
        return ternary(ret and ret > wid, ret, nil)
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
