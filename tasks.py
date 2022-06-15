
from builders import SET_MODEL_BUILDERS, CONV_MODEL_BUILDERS
from trainer import Trainer, CountingTrainer, MetaDatasetTrainer
from generators import MetaDatasetGenerator, OmniglotCooccurenceGenerator, MN

class Task():
    def __init__(self, args):
        self.args = args

    def build_model(self, pretrained_model=None):
        return SET_MODEL_BUILDERS[self.args.model](self.args)
    
    def build_dataset(self):
        pass
    
    def build_training_args(self):
        train_args = {
            'batch_size': self.args.batch_size,
            'grad_steps': self.args.grad_steps,
            'data_kwargs': {'set_size': self.args.set_size}
        }
        eval_args = {
            'batch_size': self.args.batch_size,
            'data_kwargs': {'set_size': self.args.set_size}
        }
        return train_args, eval_args
    
    def build_trainer(self, model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, device):
        train_args, eval_args = self.build_training_args()
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'checkpoint_dir': self.args.checkpoint_dir,
            'ss_schedule': self.args.ss_schedule
        }
        trainer = Trainer(model, optimizer, train_dataset, val_dataset, test_dataset, 
            train_args, eval_args, device, scheduler=scheduler, **trainer_kwargs)
        return trainer



        

#
#   Alignment Tasks
#


class EmbeddingTask(Task):
    def build_dataset(self):
        src_emb = fasttext.load_model(os.path.join(self.args.dataset_dir, "cc.en.300.bin"))
        tgt_emb = fasttext.load_model(os.path.join(self.args.dataset_dir, "cc.fr.300.bin"))
        pairs = load_pairs(os.path.join(self.args.dataset_dir, "valid_en-fr.txt"))
        train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.1, 0.1)
        train_generator = EmbeddingAlignmentGenerator(src_emb, tgt_emb, train_pairs, device=device)
        val_generator = EmbeddingAlignmentGenerator(src_emb, tgt_emb, val_pairs, device=device)
        test_generator = EmbeddingAlignmentGenerator(src_emb, tgt_emb, test_pairs, device=device)
        return train_generator, val_generator, test_generator
    
    def build_model(self, pretrained_model=None):
        self.args.input_size=300
        return super().build_model()

class CaptionTask(Task):
    def build_dataset(self):
        if self.args.text_model == 'bert':
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokenize_fct = bert_tokenize_batch
            tokenize_args = (tokenizer,)
        elif self.args.text_model == 'ft':
            ft = fasttext.load_model(args.embed_path)
            tokenize_fct = fasttext_tokenize_batch
            tokenize_args = (ft,)

        if args.dataset == "coco":
            img_path = os.path.join(self.args.dataset_dir, "images")
            annotation_path=os.path.join(self.args.dataset_dir, "annotations")
            train_dataset, val_dataset, test_dataset = load_coco_data(img_path, annotation_path )
        else:
            img_path = os.path.join(self.args.dataset_dir, "images")
            annotation_path = os.path.join(self.args.dataset_dir, "annotations.token")
            splits_path = os.path.join(self.args.dataset_dir, "splits.json")
            train_dataset, val_dataset, test_dataset = load_flickr_data(img_path, annotation_path, splits_path)
        train_generator = CaptionGenerator(train_dataset, tokenize_fct, tokenize_args, device=device)
        val_generator = CaptionGenerator(val_dataset, tokenize_fct, tokenize_args, device=device)
        test_generator = CaptionGenerator(test_dataset, tokenize_fct, tokenize_args, device=device)
        return train_generator, val_generator, test_generator
    
    def build_model(self, pretrained_model=None):
        self.args.input_size = self.args.latent_size
        set_model = super().build_model()
        if self.args.text_model == 'bert':
            model = BertModel.from_pretrained("bert-base-uncased")
            text_encoder = BertEncoderWrapper(model)
        else:
            text_encoder = EmbeddingEncoderWrapper(self.args.embed_dim)

        if self.args.img_model == 'resnet':
            resnet = torchvision.models.resnet101(pretrained=True)
            resnet.fc = nn.Identity()
            img_encoder = ImageEncoderWrapper(resnet, 2048)
        else:
            enc = ConvEncoder.make_coco_model(256)
            img_encoder = ImageEncoderWrapper(enc, 256)
        
        return MultiSetModel(set_model, img_encoder, text_encoder)


#
#   Counting Tasks
#

class CountingTask(Task):
    def build_dataset(self):
        if self.args.dataset == "mnist":
            trainval_dataset, test_dataset = load_mnist(args.data_dir)
            n_val = int(len(trainval_dataset) * args.val_split)
            train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [len(trainval_dataset)-n_val, n_val])
            generator_cls = ImageCooccurenceGenerator
        elif self.args.dataset == "omniglot":
            train_dataset, val_dataset, test_dataset = load_omniglot(args.data_dir)
            generator_cls = OmniglotCooccurenceGenerator
            pretrain_val = train_dataset
            data_kwargs['n_chars'] = 50
        elif args.dataset == "cifar100":
            trainval_dataset, test_dataset = load_cifar(args.data_dir)
            n_val = int(len(trainval_dataset) * args.val_split)
            train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [len(trainval_dataset)-n_val, n_val])
            conv_encoder = make_resnet_model(args.latent_size)
            #conv_encoder = ConvEncoder.make_cifar_model(args.latent_size)
            n_classes=100
            generator_cls = CIFARCooccurenceGenerator
            pretrain_val = val_dataset

    def build_training_args(self):
        train_args, eval_args = super().build_trainer_args()
        if self.args.dataset == 'omniglot':
            train_args['data_kwargs']['n_chars'] = 50
            eval_args['data_kwargs']['n_chars'] = 50
        return train_args, eval_args

    def build_trainer(self, model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, device):
        train_args, eval_args = self.build_training_args()
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'checkpoint_dir': self.args.checkpoint_dir,
            'ss_schedule': self.args.ss_schedule,
            'poisson': self.poisson
        }
        trainer = CountingTrainer(model, optimizer, train_dataset, val_dataset, test_dataset, 
            train_args, eval_args, device, scheduler=scheduler, **trainer_kwargs)
        return trainer

    def build_model(self, pretrained_model=None):
        set_model = super().build_model()

        if pretrained_model == None:
            conv_encoder = CONV_MODEL_BUILDERS[self.args.dataset](self.args)
        else:
            conv_encoder = pretrained_model

        model = MultiSetImageModel(conv_encoder, set_model)
        return model


#
#   Pretraining Task
#

class ImageClassificationTask(Task):
    n_classes={
        'mnist': 10,
        'cifar': 100,
        'omniglot': -1  #fill this in later
    }
    
    def build_model(self):
        encoder = CONV_MODEL_BUILDERS[self.args.dataset](self.args)
        model = nn.Sequential(encoder, nn.Linear(self.args.latent_size, self.n_classes[self.args.dataset]))
        return model
    
    def build_trainer(self, model, optimizer, train_dataset, val_dataset, test_dataset, device):
        trainer = Pretrainer(model, optimizer, train_dataset, val_dataset, test_dataset, device, self.args.batch_size, eval_every=-1)
        return trainer


    
