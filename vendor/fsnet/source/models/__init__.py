from .fsnet import build

# set training to 'True' during training
def build_model(args, training=False):
    return build(args, training)