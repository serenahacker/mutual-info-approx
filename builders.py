from models.set import *
from models.conv import *


#
#   Set Model Builders
#

def _build_mst(args):
    model_kwargs={
        'ln':True,
        'remove_diag':False,
        'num_blocks':args.num_blocks,
        'num_heads':args.num_heads,
        'dropout':args.dropout,
        'equi':args.equi,
        'decoder_layers': args.decoder_layers,
        'merge': 'sum' if args.model == 'sum-merge' else 'concat',
    }
    set_model = MultiSetTransformer(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_msrn(args):
    model_kwargs={
        'ln':True,
        'remove_diag':False,
        'num_blocks':args.num_blocks,
        'dropout':args.dropout,
        'equi':False,
        'pool1': 'max',
        'pool2': 'max',
        'decoder_layers': args.decoder_layers
    }
    set_model = MultiRNModel(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_crossonly(args):
    model_kwargs={
        'ln':True,
        'num_blocks':args.num_blocks,
        'num_heads':args.num_heads,
        'dropout':args.dropout,
        'equi':False,
        'decoder_layers': args.decoder_layers
    }
    set_model = CrossOnlyModel(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_ssrff(args):
    model_kwargs={
        'ln':True,
        'num_blocks':args.num_blocks,
        'dropout':args.dropout,
        'equi':False,
        'decoder_layers': args.decoder_layers
    }
    set_model = NaiveRFF(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_ssrn(args):
    model_kwargs={
        'ln':True,
        'num_blocks':args.num_blocks,
        'dropout':args.dropout,
        'equi': False,
        'pool': 'max',
        'decoder_layers': args.decoder_layers
    }
    set_model = NaiveRelationNetwork(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_PINE(args):
    set_model = PINE(args.input_size, int(args.latent_size/4), 16, 2, args.hidden_size, 1)
    return set_model

def _build_sst(args):
    model_kwargs={
        'ln':True,
        'remove_diag':False,
        'num_blocks':args.num_blocks,
        'num_heads':args.num_heads,
        'dropout':args.dropout,
        'equi':False,
        'decoder_layers': args.decoder_layers
    }
    set_model = NaiveSetTransformer(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

def _build_union(args):
    model_kwargs={
        'ln':True,
        'num_blocks':args.num_blocks,
        'num_heads':args.num_heads,
        'dropout':args.dropout,
        'set_encoding': args.model == 'union-enc'
    }
    set_model = UnionTransformer(args.input_size, args.latent_size, args.hidden_size, 1, **model_kwargs)
    return set_model

SET_MODEL_BUILDERS = {
    'PINE': _build_PINE,
    'single-set-rff': _build_ssrff,
    'single-set-rn': _build_ssrn,
    'single-set-transformer': _build_sst,
    'union': _build_union,
    'union-enc': _build_union,
    'multi-set-transformer': _build_mst,
    'multi-set-rn': _build_msrn,
    'cross-only': _build_crossonly,
    'sum-merge': _build_mst
}

