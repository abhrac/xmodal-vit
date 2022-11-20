ws_root = '/home/ac1151'
photo_folder_path_train = ws_root + '/Datasets/QMUL/datasets/ChairV2/trainB'
sketch_folder_path_train = ws_root + '/Datasets/QMUL/datasets/ChairV2/trainA'
photo_folder_path_test = ws_root + '/Datasets/QMUL/datasets/ChairV2/testB'
sketch_folder_path_test = ws_root + '/Datasets/QMUL/datasets/ChairV2/testA'

teacher_encoder = 'vit_base_patch16_224'
cross_modal_fusion = 'cross-attention'
teacher_loss = 'xaqc'
xa_queue_size_teacher = 128
teacher_out_dim = 768
teacher_device = 'cuda:0'

student_encoder = 'vit_tiny_patch16_224_in21k'
student_loss = 'xaqc'
xa_queue_size_student = 256
student_embed_dim = 192
rkd = True
photo_student_device = 'cuda:0'
sketch_student_device = 'cuda:0'

teacher_tgt_dir = './checkpoints/QMUL/ChairV2/'
photo_student_tgt_dir = './checkpoints/QMUL/ChairV2/student/photo/'
sketch_student_tgt_dir = './checkpoints/QMUL/ChairV2/student/sketch/'
