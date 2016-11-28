require('paths')
require('util')
require('torch')
require('dep.feature')

home_dir = paths.dirname(paths.thisfile())
data_dir = paths.concat(home_dir, '../penntree')
out_dir = 'output/dep'
mkdirs(out_dir)

sd_dir = paths.concat(out_dir, 'penntree.sd')
sd_vocab_path = paths.concat(out_dir, 'sd-vocab.th7')
sd_action_path = paths.concat(out_dir, 'sd-actions.th7')
sd_train_path = paths.concat(out_dir, 'sd-train.th7')
sd_valid_path = paths.concat(out_dir, 'sd-valid.th7')
sd_test_path = paths.concat(out_dir, 'sd-test.th7')

lth_dir = paths.concat(out_dir, 'penntree.lth')
lth_vocab_path = paths.concat(out_dir, 'lth-vocab.th7')
lth_action_path = paths.concat(out_dir, 'lth-actions.th7')
lth_train_path = paths.concat(out_dir, 'lth-train.th7')
lth_valid_path = paths.concat(out_dir, 'lth-valid.th7')
lth_test_path = paths.concat(out_dir, 'lth-test.th7')

predefined_feature_templates = {
    chen_manning = ChenManningFeatures(),
}

rand_seed = 20151213
if rand_seed then 
    torch.manualSeed(rand_seed)
end

io.output():setvbuf("no")

root_dep_label = 'root'
