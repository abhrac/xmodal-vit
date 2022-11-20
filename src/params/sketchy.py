ws_root = '~'

photo_folder_path_train = ws_root + '/Datasets/Sketchy/rendered_256x256/256x256/photo/tx_000000000000/'
photo_folder_path_test = ws_root + '/Datasets/Sketchy/rendered_256x256/256x256/photo/test'

teacher_encoder = 'vit_base_patch16_224'
cross_modal_fusion = 'cross-attention'
teacher_loss = 'xaqc'
xa_queue_size_teacher = 64
teacher_out_dim = 768
teacher_device = 'cuda:0'

student_encoder = 'vit_small_patch16_224_in21k'
student_loss = 'xaqc'
xa_queue_size_student = 64
student_embed_dim = 384
rkd = True
photo_student_device = 'cuda:0'
sketch_student_device = 'cuda:0'

teacher_tgt_dir = './checkpoints/Sketchy/'
photo_student_tgt_dir = './checkpoints/Sketchy/student/photo/'
sketch_student_tgt_dir = './checkpoints/Sketchy/student/sketch/'
