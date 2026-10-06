      0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375,
       0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375]), 'output_session_index': Padded8Object(obj=array([1, 1])), 'output_timestamps': Padded8Object(obj=tensor([0.9807, 0.9807], dtype=torch.float64)), 'output_decoder_index': Padded8Object(obj=tensor([4, 5]))}, 'target_values': ChainObject(obj={'drifting_gratings_orientation': array([5]), 'drifting_gratings_temporal_frequency': array([2])}, allow_missing_keys=True), 'target_weights': ChainObject(obj={'drifting_gratings_orientation': array([1.], dtype=float32), 'drifting_gratings_temporal_frequency': array([1.], dtype=float32)}, allow_missing_keys=True), 'session_id': 'allen_visual_coding_ophys_2016/710504563', 'absolute_start': 32.73288, 'eval_mask': ChainObject(obj={'drifting_gratings_orientation': array([ True]), 'drifting_gratings_temporal_frequency': array([ True])}, allow_missing_keys=True)}
(Pdb) type(item)
<class 'dict'>
(Pdb) item.keys()
dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
(Pdb) item['model_inputs'].shape
*** AttributeError: 'dict' object has no attribute 'shape'
(Pdb) item['model_inputs'].keys()
dict_keys(['input_unit_index', 'input_timestamps', 'input_values', 'input_mask', 'latent_index', 'latent_timestamps', 'output_session_index', 'output_timestamps', 'output_decoder_index'])
(Pdb) item['session_id']
'allen_visual_coding_ophys_2016/710504563'
(Pdb) item['model_inputs']['output_session_index']
Padded8Object(obj=array([1, 1]))
(Pdb) item['model_inputs']['input_values'].shape
*** AttributeError: 'Padded8Object' object has no attribute 'shape'
(Pdb) item['model_inputs']['input_values'][0].shape
(527, 1)
(Pdb) item['model_inputs']['input_values'][0][:10,0]
array([ 0.15385865, -0.05583467,  0.12640716, -0.03962125,  0.15242824,
        0.2387823 , -0.16145116,  0.59976786, -0.10337107, -0.10678893],
      dtype=float32)
(Pdb) item['model_inputs']['input_unit_index'][0].shape
torch.Size([527])
(Pdb) item['model_inputs']['input_timestamps'][0].shape
(527,)
(Pdb) item['model_inputs']['input_mask'][0].shape
torch.Size([527])
(Pdb) item['model_inputs']['latent_index'][0].shape
()
(Pdb) item['model_inputs']['latent_index']
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,
        1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,
        2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,
        3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,
        4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,
        5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,
        6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,
        7,  8,  9, 10, 11, 12, 13, 14, 15])
(Pdb) item['model_inputs']['latent_timestamps'][0].shape
()
(Pdb) item['model_inputs']['latent_timestamps'][0]
0.0625
(Pdb) item['model_inputs']['latent_timestamps']
array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
       0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
       0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875,
       0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.1875,
       0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125,
       0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125, 0.3125,
       0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375,
       0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375, 0.4375,
       0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625,
       0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625, 0.5625,
       0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875,
       0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875, 0.6875,
       0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125,
       0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125, 0.8125,
       0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375,
       0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375])
(Pdb) type(item['model_inputs']['latent_timestamps'])
<class 'numpy.ndarray'>
(Pdb) type(item['model_inputs']['input_timestamps'])
<class 'torch_brain.data.collate.Padded8Object'>
(Pdb) type(item['model_inputs']['output_timestamps'])
<class 'torch_brain.data.collate.Padded8Object'>
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(220)setup_dataset_and_link_model()
-> self.val_dataset = Dataset(
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(221)setup_dataset_and_link_model()
-> root=self.cfg.data_root,
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(222)setup_dataset_and_link_model()
-> config=self.cfg.dataset,
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(223)setup_dataset_and_link_model()
-> split="valid",
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(224)setup_dataset_and_link_model()
-> transform=Compose([*eval_transforms, model.tokenize]),
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(220)setup_dataset_and_link_model()
-> self.val_dataset = Dataset(
(Pdb) n
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(226)setup_dataset_and_link_model()
-> self.val_dataset.disable_data_leakage_check()
(Pdb) n
[2026-01-22 23:54:50,930][root][WARNING] - Data leakage check is disabled. Please be absolutely sure that there is no leakage between valid and other splits.
> /pscratch/sd/a/ahhyun/EcoGFound/POYO/torch_brain/examples/poyo_plus/train.py(228)setup_dataset_and_link_model()
-> self.test_dataset = Dataset(
(Pdb) self.val_dataset[index]
*** NameError: name 'index' is not defined
(Pdb) self.val_dataset[idx]
{'model_inputs': {'input_unit_index': Padded8Object(obj=tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,  1,