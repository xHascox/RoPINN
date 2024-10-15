from models import PINN, QRes, FLS, KAN


def get_model(args):
    model_dict = {
        'PINN': PINN,
        'QRes': QRes,
        'FLS': FLS,
        'KAN': KAN,
    }
    return model_dict[args.model]