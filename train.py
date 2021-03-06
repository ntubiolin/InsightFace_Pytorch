from config import get_config
from Learner import face_learner
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet]",default='ir_se', type=str)
    parser.add_argument("-model", "--model_type", help="which modelType, [Arcface, CosFace, SphereFace]",default='ArcFace', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=4, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat, casia]",default='emore', type=str)
    parser.add_argument("-c", "--comment", help="comment to attach",default='', type=str)
    parser.add_argument("-p", "--pretrained_path", help="pretrained weight path",default='ir_se50.pth', type=str)

    # parser.add_argument("-dg", "--detach_gradient", help="Whether to detach xCos gradient",default=True, type=bool)
    # parser.add_argument("-lp", "--load_pretrained", help="Whether to load backbone pretrained weights",default=True, type=bool)
    # add detach_gradient feature
    parser.add_argument('--detach_gradient', dest='detach_gradient', action='store_true', help="Whether to detach xCos gradient")
    parser.add_argument('--no_detach_gradient', dest='detach_gradient', action='store_false')
    parser.set_defaults(detach_gradient=True)
    # add load pretrained weights feature
    parser.add_argument('--load_pretrained', dest='load_pretrained', action='store_true', help="Whether to load backbone pretrained weights")
    parser.add_argument('--no_load_pretrained', dest='load_pretrained', action='store_false')
    parser.set_defaults(load_pretrained=True)
    args = parser.parse_args()

    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth
        conf.modelType = args.model_type
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.exp_comment += args.comment
    conf.detachAttentionGradient = args.detach_gradient
    conf.usePretrainedWeights = args.load_pretrained
    conf.fixed_str = args.pretrained_path
    learner = face_learner(conf)

    learner.train(conf, args.epochs)
