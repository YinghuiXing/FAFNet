from argparse import ArgumentParser

def args_parser():

# 11-3-21-20-model4-channel_32all_withconv6_1conv5_1-e300-b32-lr0.0001-lrce100ce200-haar-allpartcc_QB

    parser = ArgumentParser()

    parser.add_argument('-bs', '--batch_size', type=int, default=32,
                        help="batch size of the data")
    parser.add_argument('-e', '--epochs', type=int, default=700, help='epoch of the train')# 500
    #parser.add_argument('--iters', type=int, default=25, help='epoch of the train')  # 376
    # data num / batch size  27 old

    # related with path  ---------------------------------------------------------
    # mat dataset path ----------------------------------------------------------------
    parser.add_argument('--data_path_mat_train', metavar='DIR', type=str, default='data/QB_allmat/train',
                        help='path to dataset')#
    parser.add_argument('--data_path_mat_val', metavar='DIR', type=str, default='data/QB_allmat/val',
                        help='path to dataset') # default='data/GF2_9mat/val'   data/QB_allmat/val  WV4_453mat WV2_allmat
    parser.add_argument('--data_path_mat_test', metavar='DIR', type=str, default='data/QB_allmat/test',
                        help='path to dataset')

    parser.add_argument('--model_path', metavar='DIR', type=str, default='model',
                        help='path for trained models')
    parser.add_argument('--result_path', type=str, default='result',
                        help='directory for results')
    parser.add_argument('--result_path_full', type=str, default='result_full',
                        help='directory for results')

    # related with  --------------------------------------------------------------
    parser.add_argument('--scale_ratio', type=int, default=4)
    # parser.add_argument('-w', '--wavename', default='haar', type=str,
    #                     help='wavename: haar, dbx, biorx.y, et al')
    parser.add_argument('-w', '--wavename', default='haar', type=str,
                        help='wavename: haar, dbx, biorx.y, et al')
    parser.add_argument('--bands', type=int, default=8, help='bands of the ms')

    # related with optimizer ----------------------------------------------------
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0003,
                        help='learning rate') # 0.0003
    # parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
    #                     metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-f', '--filter_size', default=1, type=int,
                        help='anti-aliasing filter size')
    parser.add_argument('-ld', '--lr_decay', type=str, default='step',
                        help='mode for learning rate decay')
    parser.add_argument('--step', type=int, default=30,
                        help='interval for learning rate decay in step mode')
    parser.add_argument('--schedule', type=int, nargs='+', default=[125, 200, 250],
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--warmup', action='store_true',
                        help='set lower initial learning rate to warm up the training')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    args = parser.parse_args()
    return args
 
